"""Time the vectorised Monotherapy model against the published loop version.

    python bench/bench_monotherapy.py --batch 1024 --backward                # CPU
    python bench/bench_monotherapy.py --batch 1024 --backward --device cuda  # GPU

Sweeps the bucket count, because the trade-off is real: fewer buckets means
fewer batched ops but more padding waste, and the gene-attention layer costs
O(n^2) in pathway size.

On CPU the loop version wins -- batching many small ops is a GPU optimisation
(it amortises kernel-launch overhead), and CPUs have no launch overhead to
amortise. Whether it pays off on a GPU is exactly what this script measures.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.monotherapy import MonotherapyModel
from ddprism.pathways import synthetic_gene_set
from ddprism.reference import ReferenceMonotherapyModel

NCI60_TRAIN_ROWS = 7_915_900  # Table S2 of the supplement


def make_sync(device):
    if device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def timed(fn, iters, device, warmup=2):
    """Median wall time. CUDA is asynchronous, so every sample is fenced."""
    sync = make_sync(device)
    for _ in range(warmup):
        fn()
    sync()

    samples = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def human_time(seconds):
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pathways", type=int, default=186)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--buckets", type=int, nargs="+",
                        default=[1, 8, 16, 32, 48])
    parser.add_argument("--backward", action="store_true",
                        help="time forward+backward instead of forward only")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; is the accelerator enabled?")
        return 1

    device = torch.device(args.device)
    torch.manual_seed(0)

    gene_set = synthetic_gene_set(num_pathways=args.pathways, seed=0)
    reference = ReferenceMonotherapyModel(gene_set).to(device)
    training = args.backward
    reference.train(training)

    counts = [len(genes) for genes in gene_set.values()]
    print(f"device={device}  batch={args.batch}  pathways={len(counts)}  "
          f"mode={'forward+backward' if training else 'forward'}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}  "
              f"torch={torch.__version__}")
    print(f"genes: total={sum(counts)}  median={statistics.median(counts):.0f}  "
          f"widest={max(counts)}")

    per_pathway = [torch.randn(args.batch, n, device=device) for n in counts]
    drug_fp = torch.randint(0, 2, (args.batch, 512), device=device).float()
    dose = torch.randn(args.batch, 1, device=device)

    def run_reference():
        if training:
            reference.zero_grad(set_to_none=True)
            reference([per_pathway, drug_fp, dose]).sum().backward()
        else:
            with torch.no_grad():
                reference([per_pathway, drug_fp, dose])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    reference_time = timed(run_reference, args.iters, device)
    reference_memory = peak_memory_mb(device)
    batches = NCI60_TRAIN_ROWS / args.batch

    header = (f"{'buckets':>9} {'params':>13} {'pad':>7} {'ms/batch':>10} "
              f"{'speedup':>9} {'peak MB':>9} {'NCI60 epoch':>13}")
    print(f"\n{header}\n{'-' * len(header)}")
    print(f"{'ref loop':>9} {sum(p.numel() for p in reference.parameters()):>13,} "
          f"{'1.0x':>7} {reference_time * 1000:>10.1f} {'1.0x':>9} "
          f"{reference_memory:>9.0f} {human_time(reference_time * batches):>13}")

    best = None
    for num_buckets in args.buckets:
        try:
            fast = MonotherapyModel(gene_set, num_buckets=num_buckets).to(device)
            fast.load_from_reference(reference)
            fast.train(training)
            packed = fast.pack(per_pathway)

            def run_fast():
                if training:
                    fast.zero_grad(set_to_none=True)
                    fast(packed, drug_fp, dose).sum().backward()
                else:
                    with torch.no_grad():
                        fast(packed, drug_fp, dose)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            fast_time = timed(run_fast, args.iters, device)
            speedup = reference_time / fast_time

            print(f"{num_buckets:>9} {sum(p.numel() for p in fast.parameters()):>13,} "
                  f"{fast.spec.padding_overhead:>6.1f}x {fast_time * 1000:>10.1f} "
                  f"{speedup:>8.1f}x {peak_memory_mb(device):>9.0f} "
                  f"{human_time(fast_time * batches):>13}")

            if best is None or speedup > best[1]:
                best = (num_buckets, speedup)
        except torch.cuda.OutOfMemoryError:
            print(f"{num_buckets:>9} {'out of memory -- try a smaller --batch':>60}")
            torch.cuda.empty_cache()
        finally:
            del fast, packed
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nNCI60 epoch extrapolated from {batches:,.0f} batches of {args.batch}.")
    if best:
        verdict = ("worth adopting" if best[1] > 1.2 else
                   "NOT worth adopting -- keep the reference loop")
        print(f"best: {best[0]} buckets at {best[1]:.1f}x -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
