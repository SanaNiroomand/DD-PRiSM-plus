"""Phase 1 on Kaggle: prove the model is correct on GPU, then time it.

Needs no dataset -- pathway sizes are synthetic, so this runs the moment a GPU
session starts. It answers two questions:

  1. Does the vectorised model still match the published one on CUDA?
     (The test suite only ever ran on CPU.)
  2. How much faster is it on a real GPU, and at which bucket count?

    python kaggle/gpu_check.py
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.monotherapy import MonotherapyModel
from ddprism.pathways import synthetic_gene_set
from original.ddprism_original import MonotherapyModel as ReferenceMonotherapyModel


def report_environment(device):
    print("=" * 64)
    print("ENVIRONMENT")
    print("=" * 64)
    print(f"torch      : {torch.__version__}")
    print(f"cuda build : {torch.version.cuda}")
    print(f"device     : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"gpu        : {props.name}")
        print(f"memory     : {props.total_memory / 1e9:.1f} GB")
        print(f"capability : {props.major}.{props.minor}")
    else:
        print("NO GPU FOUND -- in Kaggle, open the right-hand panel, then")
        print("Session options -> Accelerator -> GPU T4 x2, and rerun.")
    print()


def check_equivalence(device, num_pathways=48, batch=64):
    """Same check the test suite makes, but on whatever device we landed on."""
    print("=" * 64)
    print("CORRECTNESS ON THIS DEVICE")
    print("=" * 64)

    torch.manual_seed(0)
    gene_set = synthetic_gene_set(num_pathways=num_pathways, seed=0)

    # float64 on GPU is slow but exact; this is a one-off check, not a hot path.
    reference = ReferenceMonotherapyModel(gene_set).to(device).double()
    reference.train()
    for _ in range(3):
        per_pathway = [torch.randn(batch, len(g), device=device, dtype=torch.float64)
                       for g in gene_set.values()]
        drug_fp = torch.randint(0, 2, (batch, 512), device=device).double()
        dose = torch.randn(batch, 1, device=device, dtype=torch.float64)
        reference([per_pathway, drug_fp, dose])

    worst = 0.0
    for num_buckets in (1, 8, 16, num_pathways):
        fast = MonotherapyModel(gene_set, num_buckets=num_buckets).to(device).double()
        fast.load_from_reference(reference)

        for training in (False, True):
            reference.train(training)
            fast.train(training)
            per_pathway = [torch.randn(batch, len(g), device=device, dtype=torch.float64)
                           for g in gene_set.values()]
            drug_fp = torch.randint(0, 2, (batch, 512), device=device).double()
            dose = torch.randn(batch, 1, device=device, dtype=torch.float64)

            expected = reference([per_pathway, drug_fp, dose])
            actual = fast(fast.pack(per_pathway), drug_fp, dose)
            worst = max(worst, (actual - expected).abs().max().item())

        del fast
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"largest disagreement vs published model : {worst:.3e}")
    ok = worst < 1e-9
    print("verdict: MATCHES" if ok else "verdict: MISMATCH -- do not train with this")
    print()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_environment(device)

    if not args.skip_check and not check_equivalence(device):
        return 1

    print("=" * 64)
    print("SPEED")
    print("=" * 64)
    sys.argv = [
        "bench", "--device", str(device), "--batch", str(args.batch),
        "--iters", str(args.iters), "--backward",
        "--buckets", "1", "8", "16", "32", "48",
    ]
    from bench.bench_monotherapy import main as bench_main
    return bench_main()


if __name__ == "__main__":
    raise SystemExit(main())
