"""Score trained checkpoints on the held-out sets the paper reports.

Training reports on a random slice of its own trainval pool, which splits
individual measurements: the same drug on the same cell line can sit in training
at one dose and validation at another. The paper's test sets hold out whole
entities -- a pair, a cell line, a drug -- so its numbers are measured on a
strictly harder problem and the two are not comparable.

This runs the trained models over the real held-out sets. No training, one
forward pass each.

    python -m ddprism.evaluate --data data --processed processed --runs runs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.data import MonotherapyBatches
from ddprism.losses import pearson, rmse
from ddprism.train import (banner, build_data, build_monotherapy,
                           extract_monotherapy_outputs)
from original.ddprism_original import CombinationTherapyModel

# Table 2 and Figures S4-S6; the paper quotes these on the unseen pair set.
PAPER = {
    "pretrain": {"unseen_pair": (0.0830, 0.9387)},
    "finetune": {"unseen_pair": (0.0914, 0.8791)},
    "combination": {"unseen_pair": (0.0854, 0.9063)},
}

NCI60_SPLITS = ["unseen_pair", "unseen_cellline", "unseen_drug", "unseen_all"]
MONO_SPLITS = ["unseen_pair", "unseen_cellline", "unseen_drug", "unseen_all"]
COMBO_SPLITS = ["unseen_pair", "unseen_cellline", "unseen_one_drug",
                "unseen_two_drug", "unseen_all"]


def metrics(predicted, observed):
    predicted, observed = predicted.view(-1), observed.view(-1)
    residual = ((observed - predicted) ** 2).sum()
    total = ((observed - observed.mean()) ** 2).sum()
    return {"n": predicted.numel(),
            "rmse": rmse(predicted, observed).item(),
            "pcc": pearson(predicted, observed).item(),
            "r2": (1 - residual / total).item()}


def show(stage, split, result):
    published = PAPER.get(stage, {}).get(split)
    line = (f"  {split:<18} n={result['n']:>9,}   RMSE {result['rmse']:.4f}   "
            f"PCC {result['pcc']:.4f}   R2 {result['r2']:.4f}")
    if published:
        line += (f"   | paper RMSE {published[0]:.4f} PCC {published[1]:.4f}")
    print(line, flush=True)


def checkpoint_model_kind(path):
    """Which Monotherapy model wrote this checkpoint.

    The two are numerically identical but not structurally: the authors' class
    builds ``sample_multiplied_block`` and never calls it, so it appears in
    their state_dict and not in the vectorised one. Loading the wrong way round
    fails with a wall of missing-key errors that says nothing useful, so read
    the answer off the checkpoint rather than trusting a flag.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    keys = state["model"].keys()
    return "original" if any(k.startswith("sample_multiplied_block") for k in keys) else "fast"


def load_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def score_monotherapy(model, forward, data, frame, as_list, batch_size=4096):
    batches = MonotherapyBatches(
        data, frame.CELLNAME, frame.NSC, frame.CONCENTRATION, frame.VIABILITY,
        batch_size=batch_size, shuffle=False, as_list=as_list)
    predicted, observed = [], []
    for (genes, fingerprints, dose), target in batches:
        predicted.append(forward(genes, fingerprints, dose))
        observed.append(target)
    return metrics(torch.cat(predicted), torch.cat(observed))


@torch.no_grad()
def score_combination(combination, model, forward, data, frame, as_list, device):
    first = pd.DataFrame({"CELLNAME": frame.CELLNAME, "NSC": frame.NSC1,
                          "CONCENTRATION": frame.CONCENTRATION1})
    second = pd.DataFrame({"CELLNAME": frame.CELLNAME, "NSC": frame.NSC2,
                           "CONCENTRATION": frame.CONCENTRATION2})
    attention1, viability1 = extract_monotherapy_outputs(
        model, forward, data, first, device, as_list)
    attention2, viability2 = extract_monotherapy_outputs(
        model, forward, data, second, device, as_list)
    target = torch.as_tensor(frame.VIABILITY.to_numpy(np.float32),
                             device=device).view(-1, 1)

    predicted = []
    for start in range(0, len(frame), 8192):
        stop = start + 8192
        _, _, viability = combination([attention1[start:stop], attention2[start:stop],
                                       viability1[start:stop], viability2[start:stop]])
        predicted.append(viability)
    return metrics(torch.cat(predicted), target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--model", choices=["original", "fast", "auto"],
                        default="auto",
                        help="'auto' reads the kind off the checkpoint")
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(f"inputs  (device: {device})")
    gmt = next(args.data.glob("*kegg_legacy*.gmt"))
    data, gene_set, spec = build_data(args.processed, gmt, device, args.buckets)

    kind = args.model
    if kind == "auto":
        anchor = next((args.runs / s / "best.pt" for s in ("finetune", "pretrain")
                       if (args.runs / s / "best.pt").exists()), None)
        if anchor is None:
            raise SystemExit(f"no checkpoints under {args.runs} to evaluate")
        kind = checkpoint_model_kind(anchor)
    print(f"  model        : {kind} (from checkpoint)")

    model, forward, as_list = build_monotherapy(gene_set, kind,
                                                args.buckets, device)
    results = {}

    for stage, folder, splits in (
            ("pretrain", "nci60_splits", NCI60_SPLITS),
            ("finetune", "almanac_mono_splits", MONO_SPLITS)):
        best = args.runs / stage / "best.pt"
        if not best.exists():
            print(f"\n  no checkpoint at {best} -- skipping {stage}")
            continue
        banner(f"{stage}: Monotherapy model on held-out {folder}")
        load_weights(model, best, device)
        results[stage] = {}
        for split in splits:
            path = args.processed / folder / f"{split}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if "NSC1" in frame.columns:
                frame = frame.rename(columns={"NSC1": "NSC"})
            if frame.empty:
                continue
            results[stage][split] = score_monotherapy(
                model, forward, data, frame, as_list)
            show(stage, split, results[stage][split])

    best = args.runs / "combination" / "best.pt"
    if best.exists() and (args.runs / "finetune" / "best.pt").exists():
        banner("combination: Combination therapy model on held-out sets")
        load_weights(model, args.runs / "finetune" / "best.pt", device)
        combination = load_weights(CombinationTherapyModel().to(device), best, device)
        results["combination"] = {}
        for split in COMBO_SPLITS:
            path = args.processed / "almanac_combo_splits" / f"{split}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            results["combination"][split] = score_combination(
                combination, model, forward, data, frame, as_list, device)
            show("combination", split, results["combination"][split])

    banner("headline: unseen pair set, as the paper reports it")
    print(f"  {'stage':<14}{'RMSE':>9}{'paper':>9}{'PCC':>9}{'paper':>9}")
    for stage in ("pretrain", "finetune", "combination"):
        got = results.get(stage, {}).get("unseen_pair")
        if not got:
            continue
        paper_rmse, paper_pcc = PAPER[stage]["unseen_pair"]
        print(f"  {stage:<14}{got['rmse']:>9.4f}{paper_rmse:>9.4f}"
              f"{got['pcc']:>9.4f}{paper_pcc:>9.4f}")

    destination = args.out or (args.runs / "evaluation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n  -> {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
