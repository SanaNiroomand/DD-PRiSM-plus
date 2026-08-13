"""Training loop for DD-PRiSM, built around the authors' own model classes.

The models come from ``original.ddprism_original`` -- their MonotherapyModel,
their CombinationTherapyModel, unmodified. What is added here is the machinery
around them: the three-stage sequence, the schedule the paper specifies, and
checkpoint/resume, without which Kaggle's session limit costs you a whole run.

Three stages, in order:

  pretrain     MonotherapyModel on NCI60 (~8.9M rows)
  finetune     the same weights on NCI-ALMANAC monotherapy (35k rows), with
               everything frozen except the four curve-parameter heads
  combination  extract each drug's pathway attention and predicted viability
               from the frozen model, then train CombinationTherapyModel on
               ALMANAC pairs

Schedule, from the Methods section: AdamW; learning rate 1e-2, or 1e-3 when
fine-tuning; drop the rate tenfold if validation loss fails to improve by more
than 0.0005 for 10 consecutive epochs; stop after 20 such epochs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.data import MonotherapyBatches, MonotherapyTensorData
from ddprism.losses import CustomLoss, estimate_density, pearson, rmse
from ddprism.pathways import PathwaySpec, read_gmt
from original.ddprism_original import (  # the authors' classes, unmodified
    CombinationTherapyModel,
    Hook,
    MonotherapyModel,
)

THRESHOLD = 0.0005      # "did not decrease by more than a threshold value"
LR_PATIENCE = 10
STOP_PATIENCE = 20
CURVE_HEADS = ("final_y_max", "final_y_min", "final_slope", "final_IC50")


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_gene_sets(gmt_path, valid_genes):
    """KEGG pathways restricted to genes the expression matrix actually has."""
    gene_set = read_gmt(gmt_path)
    valid = set(valid_genes)
    restricted = {}
    for name, genes in gene_set.items():
        kept = [g for g in genes if g in valid]
        if kept:
            restricted[name] = kept
    return restricted


def build_data(processed, gmt_path, device, num_buckets=8, require_186=True):
    expression = pd.read_parquet(processed / "expression_zscore.parquet")
    fingerprints = pd.read_parquet(processed / "fingerprints.parquet")
    gene_set = load_gene_sets(gmt_path, expression.columns)

    # The authors' CombinationTherapyModel takes no arguments and hardcodes
    # self.num_pathway = 186, so anything else fails inside its first Linear
    # with a shape error that says nothing about the cause. Catch it here.
    if require_186 and len(gene_set) != 186:
        raise SystemExit(
            f"expected 186 KEGG pathways, got {len(gene_set)}.\n"
            f"  The published CombinationTherapyModel hardcodes 186 and will\n"
            f"  fail with a shape mismatch otherwise. Check that the .gmt is\n"
            f"  c2.cp.kegg_legacy.v2023.2.Hs and that the expression matrix\n"
            f"  still covers every pathway after the gene intersection.")

    spec = PathwaySpec(gene_set, num_buckets=num_buckets)

    print(f"  pathways     : {len(spec)}")
    print(f"  genes kept   : {sum(spec.gene_counts):,}")
    print(f"  cell lines   : {len(expression):,}")
    print(f"  fingerprints : {len(fingerprints):,}")

    by_cellline = {
        cell: [expression.loc[cell, gene_set[name]].to_numpy(dtype=np.float32)
               for name in spec.names]
        for cell in expression.index
    }
    data = MonotherapyTensorData(spec, by_cellline, fingerprints, device=device)
    data.build_pathway_list(by_cellline)
    print(f"  {data.describe()}")
    return data, gene_set, spec


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

class Run:
    """One training stage that can be stopped and resumed.

    Kaggle kills a session after roughly twelve hours, and NCI60 pretraining
    takes longer than that, so every epoch is written to disk with enough state
    to continue exactly where it left off -- weights, optimiser moments, the
    learning rate, both patience counters and the best score so far.
    """

    def __init__(self, name, model, optimizer, directory, lr):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.epoch = 0
        self.best = float("inf")
        self.since_improved = 0
        self.history = []

    @property
    def checkpoint_path(self):
        return self.directory / "checkpoint.pt"

    @property
    def best_path(self):
        return self.directory / "best.pt"

    def save(self, is_best):
        state = {
            "epoch": self.epoch,
            "best": self.best,
            "since_improved": self.since_improved,
            "lr": self.lr,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "torch_rng": torch.get_rng_state(),
        }
        # Write beside the target then rename, so an interrupted save cannot
        # leave a half-written checkpoint where a good one used to be.
        staging = self.checkpoint_path.with_suffix(".tmp")
        torch.save(state, staging)
        staging.replace(self.checkpoint_path)
        if is_best:
            torch.save({"epoch": self.epoch, "model": self.model.state_dict()},
                       self.best_path)
        (self.directory / "history.json").write_text(
            json.dumps(self.history, indent=1), encoding="utf-8")

    def resume(self):
        if not self.checkpoint_path.exists():
            return False
        state = torch.load(self.checkpoint_path, map_location="cpu",
                           weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]
        self.best = state["best"]
        self.since_improved = state["since_improved"]
        self.lr = state["lr"]
        self.history = state.get("history", [])
        torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        print(f"  resumed {self.name} at epoch {self.epoch} "
              f"(best {self.best:.6f}, lr {self.lr:.2e})")
        return True

    def observe(self, validation_loss):
        """Apply the paper's schedule. Returns True when training should stop."""
        improved = validation_loss < self.best - THRESHOLD
        if improved:
            self.best = validation_loss
            self.since_improved = 0
        else:
            self.since_improved += 1

        if self.since_improved and self.since_improved % LR_PATIENCE == 0:
            self.lr /= 10
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr
            print(f"    no improvement for {self.since_improved} epochs "
                  f"-> lr {self.lr:.2e}")

        return improved, self.since_improved >= STOP_PATIENCE


# --------------------------------------------------------------------------
# monotherapy stages
# --------------------------------------------------------------------------

def run_monotherapy_epoch(model, batches, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals, predictions, actuals = [], [], []

    for (genes, fingerprints, dose), target in batches:
        prediction = model([genes, fingerprints, dose])
        loss, _, _ = loss_fn(prediction, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        totals.append(loss.detach())
        predictions.append(prediction.detach())
        actuals.append(target)

    predicted = torch.cat(predictions).view(-1)
    observed = torch.cat(actuals).view(-1)
    return (torch.stack(totals).mean().item(),
            rmse(predicted, observed).item(),
            pearson(predicted, observed).item())


def train_monotherapy(model, train_batches, validation_batches, loss_fn,
                      directory, lr, max_epochs, deadline, stage):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    run = Run(stage, model, optimizer, directory, lr)
    run.resume()

    while run.epoch < max_epochs:
        run.epoch += 1
        started = time.time()
        train_loss, train_rmse, train_pcc = run_monotherapy_epoch(
            model, train_batches, loss_fn, optimizer)
        with torch.no_grad():
            val_loss, val_rmse, val_pcc = run_monotherapy_epoch(
                model, validation_batches, loss_fn)

        improved, should_stop = run.observe(val_loss)
        run.history.append({"epoch": run.epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "val_rmse": val_rmse,
                            "val_pcc": val_pcc, "lr": run.lr})
        run.save(is_best=improved)

        print(f"  epoch {run.epoch:>3}  train {train_loss:.5f}  "
              f"val {val_loss:.5f}  RMSE {val_rmse:.4f}  PCC {val_pcc:.4f}  "
              f"{time.time() - started:.0f}s{'  *' if improved else ''}",
              flush=True)

        if should_stop:
            print(f"  early stop: no improvement for {STOP_PATIENCE} epochs")
            break
        if deadline and time.time() > deadline:
            print("  time budget reached -- checkpoint saved, rerun to continue")
            return run, False

    return run, True


def freeze_for_finetuning(model):
    """Unfreeze only the four curve-parameter heads.

    The paper: "we froze most model parameters to preserve the model's ability
    to infer cancer-drug interactions through biological pathways. Only the
    final layers that predict the four parameters of the response curve were
    unfrozen, allowing adjustment for batch effects between datasets."
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for head in CURVE_HEADS:
        for parameter in getattr(model, head).parameters():
            parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} of {total:,} parameters "
          f"({trainable / total * 100:.2f}%)")
    return model


# --------------------------------------------------------------------------
# combination stage
# --------------------------------------------------------------------------

@torch.no_grad()
def extract_monotherapy_outputs(model, data, frame, device, batch_size=4096):
    """Pathway attention and predicted viability for each (cell, drug, dose).

    The authors' forward returns viability only; the attention is recovered with
    a forward hook on ``sample_attention_block``, which is how their own
    pipeline does it.
    """
    model.eval()
    hook = Hook(model.sample_attention_block)

    cell_rows = torch.as_tensor([data.cell_index[c] for c in frame.CELLNAME],
                                dtype=torch.long, device=device)
    drug_rows = torch.as_tensor([data.drug_index[d] for d in frame.NSC],
                                dtype=torch.long, device=device)
    dose = torch.as_tensor(frame.CONCENTRATION.to_numpy(np.float32),
                           device=device).view(-1, 1)

    attentions, viabilities = [], []
    for start in range(0, len(frame), batch_size):
        stop = start + batch_size
        genes, fingerprints = data.gather_list(cell_rows[start:stop],
                                               drug_rows[start:stop])
        viability = model([genes, fingerprints, dose[start:stop]])
        attentions.append(hook.o.detach())
        viabilities.append(viability.detach())

    hook.close()
    return torch.cat(attentions), torch.cat(viabilities)


def run_combination_epoch(model, tensors, loss_fn, batch_size,
                          optimizer=None, generator=None):
    attention1, attention2, viability1, viability2, target = tensors
    training = optimizer is not None
    model.train(training)

    count = len(target)
    order = (torch.randperm(count, generator=generator, device=target.device)
             if training else torch.arange(count, device=target.device))

    totals, predictions, actuals = [], [], []
    for start in range(0, count, batch_size):
        pick = order[start:start + batch_size]
        if len(pick) < 2:            # BatchNorm needs more than one sample
            continue
        _, _, prediction = model([attention1[pick], attention2[pick],
                                  viability1[pick], viability2[pick]])
        loss, _, _ = loss_fn(prediction, target[pick])
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        totals.append(loss.detach())
        predictions.append(prediction.detach())
        actuals.append(target[pick])

    predicted = torch.cat(predictions).view(-1)
    observed = torch.cat(actuals).view(-1)
    return (torch.stack(totals).mean().item(),
            rmse(predicted, observed).item(),
            pearson(predicted, observed).item())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def split_trainval(frame, fraction=1 / 9, seed=0):
    """The paper splits its training-validation pool 8:1."""
    generator = np.random.default_rng(seed)
    mask = generator.random(len(frame)) < fraction
    return frame[~mask], frame[mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="raw directory, for the KEGG .gmt")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--stage", default="all",
                        choices=["all", "pretrain", "finetune", "combination"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--max-hours", type=float, default=10.5,
                        help="stop cleanly before Kaggle kills the session")
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None

    banner(f"inputs  (device: {device})")
    gmt = next(args.data.glob("*kegg_legacy*.gmt"))
    data, gene_set, spec = build_data(args.processed, gmt, device, args.buckets)

    stages = (["pretrain", "finetune", "combination"]
              if args.stage == "all" else [args.stage])

    model = MonotherapyModel(gene_set).to(device)

    if "pretrain" in stages:
        banner("stage 1/3  pretrain on NCI60")
        frame = pd.read_parquet(args.processed / "nci60_splits" / "trainval.parquet")
        train_frame, val_frame = split_trainval(frame, seed=args.seed)
        print(f"  train {len(train_frame):,}   validation {len(val_frame):,}")

        density = estimate_density(train_frame.VIABILITY.to_numpy())
        loss_fn = CustomLoss(density).to(device)

        as_list = True
        train_batches = MonotherapyBatches(
            data, train_frame.CELLNAME, train_frame.NSC, train_frame.CONCENTRATION,
            train_frame.VIABILITY, batch_size=args.batch_size, as_list=as_list)
        val_batches = MonotherapyBatches(
            data, val_frame.CELLNAME, val_frame.NSC, val_frame.CONCENTRATION,
            val_frame.VIABILITY, batch_size=args.batch_size, shuffle=False,
            as_list=as_list)

        run, finished = train_monotherapy(
            model, train_batches, val_batches, loss_fn,
            args.out / "pretrain", 1e-2, args.max_epochs, deadline, "pretrain")
        if not finished:
            return 0

    if "finetune" in stages:
        banner("stage 2/3  fine-tune on NCI-ALMANAC monotherapy")
        best = args.out / "pretrain" / "best.pt"
        if best.exists():
            model.load_state_dict(torch.load(best, map_location=device,
                                             weights_only=False)["model"])
            print(f"  loaded pretrained weights from {best}")
        freeze_for_finetuning(model)

        frame = pd.read_parquet(args.processed / "almanac_mono.parquet")
        frame = frame.rename(columns={"NSC1": "NSC"})
        train_frame, val_frame = split_trainval(frame, seed=args.seed)
        print(f"  train {len(train_frame):,}   validation {len(val_frame):,}")

        density = estimate_density(train_frame.VIABILITY.to_numpy())
        loss_fn = CustomLoss(density).to(device)

        train_batches = MonotherapyBatches(
            data, train_frame.CELLNAME, train_frame.NSC, train_frame.CONCENTRATION,
            train_frame.VIABILITY, batch_size=args.batch_size, as_list=True)
        val_batches = MonotherapyBatches(
            data, val_frame.CELLNAME, val_frame.NSC, val_frame.CONCENTRATION,
            val_frame.VIABILITY, batch_size=args.batch_size, shuffle=False,
            as_list=True)

        run, finished = train_monotherapy(
            model, train_batches, val_batches, loss_fn,
            args.out / "finetune", 1e-3, args.max_epochs, deadline, "finetune")
        if not finished:
            return 0

    if "combination" in stages:
        banner("stage 3/3  combination therapy model")
        best = args.out / "finetune" / "best.pt"
        if best.exists():
            model.load_state_dict(torch.load(best, map_location=device,
                                             weights_only=False)["model"])
            print(f"  loaded fine-tuned weights from {best}")

        combo = pd.read_parquet(args.processed / "almanac_combo.parquet")
        print(f"  combination rows: {len(combo):,}")

        print("  extracting pathway attention from the frozen Monotherapy model")
        first = pd.DataFrame({"CELLNAME": combo.CELLNAME, "NSC": combo.NSC1,
                              "CONCENTRATION": combo.CONCENTRATION1})
        second = pd.DataFrame({"CELLNAME": combo.CELLNAME, "NSC": combo.NSC2,
                               "CONCENTRATION": combo.CONCENTRATION2})
        attention1, viability1 = extract_monotherapy_outputs(model, data, first, device)
        attention2, viability2 = extract_monotherapy_outputs(model, data, second, device)
        target = torch.as_tensor(combo.VIABILITY.to_numpy(np.float32),
                                 device=device).view(-1, 1)

        indices = np.arange(len(combo))
        generator = np.random.default_rng(args.seed)
        held = generator.random(len(combo)) < 1 / 9
        train_idx = torch.as_tensor(indices[~held], device=device)
        val_idx = torch.as_tensor(indices[held], device=device)
        print(f"  train {len(train_idx):,}   validation {len(val_idx):,}")

        density = estimate_density(combo.VIABILITY.to_numpy()[~held])
        loss_fn = CustomLoss(density).to(device)

        combination = CombinationTherapyModel().to(device)
        optimizer = torch.optim.AdamW(combination.parameters(), lr=1e-2)
        run = Run("combination", combination, optimizer,
                  args.out / "combination", 1e-2)
        run.resume()

        pack = lambda idx: (attention1[idx], attention2[idx],
                            viability1[idx], viability2[idx], target[idx])
        train_tensors, val_tensors = pack(train_idx), pack(val_idx)

        while run.epoch < args.max_epochs:
            run.epoch += 1
            started = time.time()
            train_loss, _, _ = run_combination_epoch(
                combination, train_tensors, loss_fn, args.batch_size, optimizer)
            with torch.no_grad():
                val_loss, val_rmse, val_pcc = run_combination_epoch(
                    combination, val_tensors, loss_fn, args.batch_size)

            improved, should_stop = run.observe(val_loss)
            run.history.append({"epoch": run.epoch, "train_loss": train_loss,
                                "val_loss": val_loss, "val_rmse": val_rmse,
                                "val_pcc": val_pcc, "lr": run.lr})
            run.save(is_best=improved)
            print(f"  epoch {run.epoch:>3}  train {train_loss:.5f}  "
                  f"val {val_loss:.5f}  RMSE {val_rmse:.4f}  PCC {val_pcc:.4f}  "
                  f"{time.time() - started:.0f}s{'  *' if improved else ''}",
                  flush=True)

            if should_stop:
                print(f"  early stop: no improvement for {STOP_PATIENCE} epochs")
                break
            if deadline and time.time() > deadline:
                print("  time budget reached -- checkpoint saved, rerun to continue")
                return 0

        print(f"\n  paper reports RMSE 0.0854, PCC 0.9063 on the unseen pair set")

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
