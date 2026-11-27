#!/usr/bin/env python
"""Fair, multi-seed comparison of drug representations on the SAME drug set.

Every configuration (Morgan, Morgan+L1000, L1000-only) is restricted to the
L1000-covered drugs via --restrict-to-source l1000, so all three are trained and
scored on one identical universe. Each is run over several seeds; we report
mean +/- std per split. Inputs (01/02/02b outputs) are discovered under
/kaggle/input; results land in /kaggle/working/fair_multiseed.json.
"""
import os, sys, glob, json, subprocess, statistics

SEEDS = [0, 1, 2]
MAX_HOURS = os.environ.get("FAIR_MAX_HOURS", "0.5")
CONFIGS = [
    ("morgan",       "morgan",       []),
    ("morgan-l1000", "morgan+l1000", ["--drug-feature-fill", "zero"]),
    ("l1000",        "l1000",        []),
]

def one(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise SystemExit(f"required input not found: {pattern}")
    return hits[0]

DATA = os.path.dirname(one("DOSERESP.zip"))
PROC_SRC = os.path.dirname(one("nci60_filtered.parquet"))
L1000 = one("drug_signatures_l1000.parquet")

# Merge the preprocessed tables and the L1000 parquet into one PROCESSED dir
MERGED = "/kaggle/working/merged"
os.makedirs(MERGED, exist_ok=True)
for name in os.listdir(PROC_SRC):
    dst = os.path.join(MERGED, name)
    if not os.path.exists(dst):
        os.symlink(os.path.join(PROC_SRC, name), dst)
l1dst = os.path.join(MERGED, "drug_signatures_l1000.parquet")
if not os.path.exists(l1dst):
    os.symlink(L1000, l1dst)
print("DATA    :", DATA, flush=True)
print("PROCESSED:", MERGED, "(+ l1000)", flush=True)

RUNROOT = "/kaggle/working/runs"
results = {}
for tag, feat, extra in CONFIGS:
    results[tag] = {}
    for seed in SEEDS:
        out = f"{RUNROOT}/{tag}_s{seed}"
        print(f"\n===== TRAIN {tag} seed {seed} =====", flush=True)
        train = ["python", "-m", "ddprism.train", "--data", DATA,
                 "--processed", MERGED, "--out", out, "--drug-features", feat,
                 "--restrict-to-source", "l1000", "--seed", str(seed),
                 "--curve-width", "2", "--batch-size", "1024",
                 "--max-hours", MAX_HOURS] + extra
        subprocess.run(train, check=True)
        subprocess.run(["python", "-m", "ddprism.evaluate", "--data", DATA,
                        "--processed", MERGED, "--runs", out], check=True)
        ev = os.path.join(out, "evaluation.json")
        results[tag][seed] = json.load(open(ev)) if os.path.exists(ev) else None

# Aggregate mean/std per config, stage, split, metric
summary = {}
for tag, byseed in results.items():
    runs = [r for r in byseed.values() if r]
    agg = {}
    for stage in set().union(*[r.keys() for r in runs]) if runs else []:
        agg[stage] = {}
        splits = set().union(*[r.get(stage, {}).keys() for r in runs])
        for sp in splits:
            agg[stage][sp] = {}
            for metric in ("pcc", "rmse"):
                vals = [r[stage][sp][metric] for r in runs
                        if sp in r.get(stage, {}) and metric in r[stage][sp]]
                if vals:
                    agg[stage][sp][metric] = {
                        "mean": statistics.mean(vals),
                        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals)}
    summary[tag] = agg

out_path = "/kaggle/working/fair_multiseed.json"
json.dump({"seeds": SEEDS, "max_hours": MAX_HOURS,
           "results": results, "summary": summary},
          open(out_path, "w"), indent=1)

# Readable table
for tag in summary:
    print(f"\n================ {tag} ================", flush=True)
    for stage in sorted(summary[tag]):
        print(f"  [{stage}]")
        for sp in sorted(summary[tag][stage]):
            m = summary[tag][stage][sp].get("pcc")
            if m:
                print(f"    {sp:<18} PCC {m['mean']:.3f} +/- {m['std']:.3f}  (n={m['n']})")
print("\nwrote", out_path, flush=True)
