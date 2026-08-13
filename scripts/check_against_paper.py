"""Check our understanding of the model against the paper's own published output.

Supplementary Data 3 is not a summary -- it is 2,556 rows of the trained model's
actual output, with the decomposition laid bare: both monotherapy predictions,
both coefficients, the synergy term, and the combined viability. That makes it a
ground truth for two things a reimplementation can otherwise only assume.

    python scripts/check_against_paper.py --supp path/to/supplementarydata3_bbae717.xlsx

Findings recorded here, both reproducible from the file:

1. E = alpha*E1 + beta*E2 + gamma holds to 1.5e-07, and alpha + beta = 1. So the
   combination equation in ddprism matches theirs exactly, including the softmax
   over the two monotherapy coefficients.

2. Predicted viability runs down to -0.78. Viability is the surviving fraction
   of cells, so negative values are physically impossible. This confirms a bug
   visible in the published source: CombinationTherapyModel builds
   `efficacy_relu` and `viability_relu` in __init__ and never calls either in
   forward, leaving the output unbounded. It is in their released results, not
   just their code.

   The upper bound of 0.30 is not a bug -- Supplementary Data 3 is filtered to
   combinations the paper calls sensitive, defined as viability below 30%.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load(path):
    book = pd.ExcelFile(path)
    frames = []
    for sheet in book.sheet_names:
        frame = book.parse(sheet)
        frame["CANCERTYPE"] = sheet
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_decomposition(frame, tolerance=1e-5):
    """E = alpha*E1 + beta*E2 + gamma, with E = 1 - viability."""
    efficacy = 1 - frame.PREDICTED_VIABILITY
    rebuilt = (frame.COEFFICIENT1 * (1 - frame.PREDICTED_VIABILITY1)
               + frame.COEFFICIENT2 * (1 - frame.PREDICTED_VIABILITY2)
               + frame.SYNERGY)
    error = (efficacy - rebuilt).abs().max()
    ok = error < tolerance
    print(f"  decomposition identity   max error {error:.3e}   "
          f"{'HOLDS' if ok else 'FAILS'}")
    return ok


def check_softmax(frame, tolerance=1e-5):
    error = (frame.COEFFICIENT1 + frame.COEFFICIENT2 - 1).abs().max()
    ok = error < tolerance
    print(f"  alpha + beta == 1        max error {error:.3e}   "
          f"{'HOLDS' if ok else 'FAILS'}")
    return ok


def report_bounds(frame):
    low, high = frame.PREDICTED_VIABILITY.min(), frame.PREDICTED_VIABILITY.max()
    negative = int((frame.PREDICTED_VIABILITY < 0).sum())
    print(f"  predicted viability      [{low:.4f}, {high:.4f}]")
    print(f"  physically impossible    {negative:,} of {len(frame):,} rows "
          f"below zero ({negative / len(frame) * 100:.1f}%)")
    if negative:
        print("    ^ unbounded output: the published CombinationTherapyModel")
        print("      defines efficacy_relu and viability_relu but never calls them")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supp", type=Path, required=True,
                        help="supplementarydata3_bbae717.xlsx from the paper")
    args = parser.parse_args()

    if not args.supp.exists():
        print(f"not found: {args.supp}")
        return 2

    frame = load(args.supp)
    print(f"Supplementary Data 3: {len(frame):,} rows across "
          f"{frame.CANCERTYPE.nunique()} cancer types\n")

    ok = check_decomposition(frame)
    ok = check_softmax(frame) and ok
    report_bounds(frame)

    print("\nverdict:", "our combination equation matches the paper"
          if ok else "MISMATCH -- the equation in ddprism is not theirs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
