"""Measure the curve-prediction bottleneck on a trained checkpoint.

The published model ends ``Linear(8, 2) -> BatchNorm1d(2) -> ReLU``, and reads
all four Hill-curve parameters off that one vector with four ``Linear(2, 1)``
heads. Two dimensions cannot carry four independent numbers, and the ReLU
leaves roughly half of what there is at zero.

This measures the consequence rather than asserting it:

  * **effective rank** of the four predicted parameters over a batch of real
    inputs -- how many independent directions they actually span. 4 is the
    ideal, 1 means the four parameters are one number wearing four hats.
  * **dead fraction** of the bottleneck -- units the ReLU zeroes for every
    sample in the batch, which are not merely inactive but useless.
  * the **correlation matrix** of the four parameters, which shows whether
    y_max and y_min are free to move apart at all.

Run it on the baseline and on a widened run; if widening the head did not move
the rank, it did not fix anything and should not be reported as having done so.

    python scripts/measure_curve_rank.py --data data --processed processed \\
        --runs runs --stage pretrain
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.data import MonotherapyBatches  # noqa: E402
from ddprism.evaluate import (checkpoint_curve_width,  # noqa: E402
                              checkpoint_model_kind, run_config)
from ddprism.train import banner, build_data, build_monotherapy  # noqa: E402

CURVE_HEADS = ("final_y_max", "final_y_min", "final_slope", "final_IC50")
LABELS = ("y_max", "y_min", "slope", "IC50")


def effective_rank(matrix, tolerance=0.01):
    """Singular values above `tolerance` of the largest, after centring.

    Centring matters: four parameters that differ only by a constant offset
    are still one degree of freedom, and an uncentred SVD would score that as
    rank 2.
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[0] <= 0:
        return 0, singular
    return int((singular / singular[0] > tolerance).sum()), singular


@torch.no_grad()
def probe(model, forward, data, frame, device, as_list, batch_size=4096):
    """Bottleneck activations and the four curve parameters, on real inputs."""
    captured = {}
    handle = model.final_block.register_forward_hook(
        lambda _m, _i, output: captured.__setitem__("bottleneck", output))

    model.eval()
    batches = MonotherapyBatches(
        data, frame.CELLNAME, frame.NSC, frame.CONCENTRATION, frame.VIABILITY,
        batch_size=batch_size, shuffle=False, as_list=as_list)

    bottlenecks, parameters = [], []
    for index, ((genes, fingerprints, dose), _) in enumerate(batches):
        forward(genes, fingerprints, dose)
        embed = captured["bottleneck"]
        bottlenecks.append(embed.cpu())
        parameters.append(torch.cat(
            [getattr(model, name)(embed) for name in CURVE_HEADS], dim=1).cpu())
        if index >= 2:                      # three batches is plenty
            break

    handle.remove()
    return torch.cat(bottlenecks).numpy(), torch.cat(parameters).numpy()


def report(bottleneck, parameters):
    width = bottleneck.shape[1]
    rank, singular = effective_rank(parameters)

    always_zero = int((bottleneck <= 0).all(axis=0).sum())
    zero_fraction = float((bottleneck <= 0).mean())

    print(f"  samples probed                 : {len(bottleneck):,}")
    print(f"  bottleneck width               : {width}")
    print(f"  rank of the 4 curve parameters : {rank} of 4"
          f"{'   <-- one number wearing four hats' if rank <= 1 else ''}")
    print(f"  singular values (normalised)   : "
          + "  ".join(f"{s / singular[0]:.3f}" for s in singular))
    print(f"  units dead for every sample    : {always_zero} of {width}")
    print(f"  fraction of activations at zero: {zero_fraction:.0%}")

    print("\n  correlation between the four parameters:")
    correlation = np.corrcoef(parameters, rowvar=False)
    print("            " + "".join(f"{name:>9}" for name in LABELS))
    for row, name in enumerate(LABELS):
        print(f"  {name:>8}  " + "".join(f"{correlation[row, col]:>9.3f}"
                                         for col in range(4)))

    inverted = int((parameters[:, 0] < parameters[:, 1]).sum())
    print(f"\n  samples with y_max < y_min     : {inverted:,} of "
          f"{len(parameters):,} ({inverted / len(parameters):.0%})")
    print("    Nothing in the model constrains the curve's top to sit above")
    print("    its bottom; where this is non-zero the fitted curve is upside")
    print("    down.")
    return {"width": width, "rank": rank, "dead_units": always_zero,
            "zero_fraction": zero_fraction, "inverted": inverted}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--stage", default="pretrain",
                        choices=["pretrain", "finetune"])
    parser.add_argument("--split", default="unseen_pair")
    parser.add_argument("--buckets", type=int, default=8)
    args = parser.parse_args()

    checkpoint = args.runs / args.stage / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = run_config(args.runs)

    banner(f"curve bottleneck: {args.runs.name}/{args.stage}  (device: {device})")
    data, gene_set, _ = build_data(
        args.processed, next(args.data.glob("*kegg_legacy*.gmt")), device,
        args.buckets, drug_features=config.get("drug_features", "morgan"),
        standardize=config.get("standardize", True))

    model, forward, as_list = build_monotherapy(
        gene_set, checkpoint_model_kind(checkpoint), args.buckets, device,
        drug_dim=data.drug_dim, curve_width=checkpoint_curve_width(checkpoint),
        curve_relu=config.get("curve_relu", True))
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=False)["model"])

    folder = "nci60_splits" if args.stage == "pretrain" else "almanac_mono_splits"
    frame = pd.read_parquet(args.processed / folder / f"{args.split}.parquet")
    if "NSC1" in frame.columns:
        frame = frame.rename(columns={"NSC1": "NSC"})

    banner(f"measured on {folder}/{args.split}")
    report(*probe(model, forward, data, frame, device, as_list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
