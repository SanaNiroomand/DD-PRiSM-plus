"""End-to-end smoke test for the training loop, on tiny synthetic data.

The point is not accuracy -- it is that all three stages run against the
authors' model classes, that freezing does what the paper describes, and above
all that a stage can be interrupted and resumed. Kaggle terminates sessions
after roughly twelve hours and NCI60 pretraining takes longer, so resume is not
a nicety; if it is broken, every run starts from scratch.

Run with:  python -m pytest tests/test_train.py -v
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddprism.train import CURVE_HEADS, Run, freeze_for_finetuning, split_trainval
from original.ddprism_original import MonotherapyModel


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A miniature stand-in for the preprocessing output."""
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("tiny")
    raw, processed = root / "raw", root / "processed"
    (processed / "nci60_splits").mkdir(parents=True)
    raw.mkdir(parents=True)

    # 186 pathways, because the authors' CombinationTherapyModel takes no
    # arguments and hardcodes self.num_pathway = 186. Anything else dies inside
    # its first Linear. Kept tiny (4-10 genes each) so the test stays quick.
    genes = [f"G{i}" for i in range(400)]
    cells = [f"CELL{i}" for i in range(6)]
    drugs = list(range(20))

    picker = np.random.default_rng(1)
    lines = []
    for p in range(186):
        size = int(picker.integers(4, 10))
        start = int(picker.integers(0, len(genes) - size))
        members = genes[start:start + size]
        lines.append("\t".join([f"KEGG_P{p:03d}", "na", *members]))
    (raw / "c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    pd.DataFrame(rng.standard_normal((len(cells), len(genes))).astype(np.float32),
                 index=cells, columns=genes).to_parquet(
        processed / "expression_zscore.parquet")

    fingerprints = pd.DataFrame(
        rng.integers(0, 2, (len(drugs), 512)).astype(np.uint8),
        index=pd.Index(drugs, name="NSC"))
    fingerprints.columns = [str(c) for c in fingerprints.columns]
    fingerprints.to_parquet(processed / "fingerprints.parquet")

    def responses(n):
        return pd.DataFrame({
            "CELLNAME": rng.choice(cells, n),
            "NSC": rng.choice(drugs, n),
            "CONCENTRATION": rng.uniform(-2, 1, n).astype(np.float32),
            "VIABILITY": rng.uniform(0, 1.2, n).astype(np.float32)})

    responses(256).to_parquet(processed / "nci60_splits" / "trainval.parquet")
    responses(192).to_parquet(processed / "almanac_mono.parquet")

    pd.DataFrame({
        "CELLNAME": rng.choice(cells, 192),
        "NSC1": rng.choice(drugs, 192),
        "NSC2": rng.choice(drugs, 192),
        "CONCENTRATION1": rng.uniform(-2, 1, 192).astype(np.float32),
        "CONCENTRATION2": rng.uniform(-2, 1, 192).astype(np.float32),
        "VIABILITY": rng.uniform(0, 1.2, 192).astype(np.float32),
    }).to_parquet(processed / "almanac_combo.parquet")

    return raw, processed


def train(raw, processed, out, extra=()):
    command = [sys.executable, "-m", "ddprism.train",
               "--data", str(raw), "--processed", str(processed),
               "--out", str(out), "--batch-size", "64", *extra]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("model", ["original", "fast"])
def test_all_three_stages_run(tiny, tmp_path, model):
    """Both model paths must carry the full pipeline, not just the default."""
    raw, processed = tiny
    result = train(raw, processed, tmp_path / "runs",
                   ("--max-epochs", "2", "--model", model))
    assert result.returncode == 0, result.stdout + result.stderr

    for stage in ("pretrain", "finetune", "combination"):
        assert (tmp_path / "runs" / stage / "best.pt").exists(), f"{stage} produced no best.pt"
        assert (tmp_path / "runs" / stage / "checkpoint.pt").exists()
    assert "stage 3/3" in result.stdout


def test_resume_continues_instead_of_restarting(tiny, tmp_path):
    """The whole point of checkpointing: epoch numbers must carry over."""
    raw, processed = tiny
    out = tmp_path / "runs"

    first = train(raw, processed, out, ("--stage", "pretrain", "--max-epochs", "2"))
    assert first.returncode == 0, first.stdout + first.stderr
    assert "epoch   1" in first.stdout and "epoch   2" in first.stdout

    second = train(raw, processed, out, ("--stage", "pretrain", "--max-epochs", "4"))
    assert second.returncode == 0, second.stdout + second.stderr

    assert "resumed pretrain at epoch 2" in second.stdout
    assert "epoch   3" in second.stdout and "epoch   4" in second.stdout
    # It must not redo work already done.
    assert "epoch   1" not in second.stdout


def test_checkpoint_carries_optimizer_and_schedule_state(tiny, tmp_path):
    raw, processed = tiny
    out = tmp_path / "runs"
    train(raw, processed, out, ("--stage", "pretrain", "--max-epochs", "2"))

    state = torch.load(out / "pretrain" / "checkpoint.pt", map_location="cpu",
                       weights_only=False)
    for key in ("epoch", "best", "since_improved", "lr", "model", "optimizer"):
        assert key in state, f"checkpoint is missing {key}"
    assert state["epoch"] == 2
    assert state["optimizer"]["state"], "optimiser moments were not saved"


def test_finetuning_unfreezes_only_the_curve_heads(tiny):
    _, processed = tiny
    gene_set = {f"P{i}": [f"g{i}_{j}" for j in range(6)] for i in range(8)}
    model = freeze_for_finetuning(MonotherapyModel(gene_set))

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable, "nothing is trainable"
    for name in trainable:
        assert name.split(".")[0] in CURVE_HEADS, f"{name} should be frozen"
    # All four heads participate.
    assert {n.split(".")[0] for n in trainable} == set(CURVE_HEADS)


def test_schedule_matches_the_paper():
    """LR drops after 10 stale epochs; training stops after 20."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    run = Run("t", model, optimizer, Path("."), 1e-2)
    run.best = 1.0

    for epoch in range(1, 10):
        _, stop = run.observe(1.0)      # never improves
        assert not stop
    assert run.lr == pytest.approx(1e-2)

    _, stop = run.observe(1.0)          # tenth stale epoch
    assert run.lr == pytest.approx(1e-3)
    assert not stop

    for _ in range(10):
        _, stop = run.observe(1.0)
    assert stop, "should stop after 20 stale epochs"


def test_improvement_must_exceed_the_threshold():
    model = torch.nn.Linear(2, 1)
    run = Run("t", model, torch.optim.AdamW(model.parameters()), Path("."), 1e-2)
    run.best = 1.0

    improved, _ = run.observe(1.0 - 0.0001)   # smaller than THRESHOLD
    assert not improved
    improved, _ = run.observe(1.0 - 0.01)     # comfortably larger
    assert improved


def test_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame({"x": range(900)})
    a1, b1 = split_trainval(frame, seed=7)
    a2, b2 = split_trainval(frame, seed=7)

    assert len(a1) + len(b1) == len(frame)
    assert not set(a1.index) & set(b1.index)
    assert list(a1.index) == list(a2.index)
    assert 0.06 < len(b1) / len(frame) < 0.17     # nominally 1/9
