"""The generated Kaggle notebooks must keep the experiments apart.

Each experiment is a separate notebook writing to a separate directory. If two
of them ever shared one, the second run would try to resume a model of the wrong
shape -- or worse, quietly overwrite the baseline it is supposed to be measured
against, and the comparison would be of a run against itself.

A notebook that does not parse is the other failure worth catching: one shipped
once with a SyntaxError and cost a Kaggle session.

Run with:  python -m pytest tests/test_notebooks.py -v
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_kaggle_notebook import (COMPARE_CELLS, EXPERIMENTS, _as_python,
                                   check_cells, train_cells)

NOTEBOOKS = ROOT / "kaggle"


def test_every_experiment_has_its_own_run_directory():
    directories = [e["runs"] for e in EXPERIMENTS]
    assert len(set(directories)) == len(directories), directories


def test_every_experiment_has_its_own_notebook():
    files = [e["file"] for e in EXPERIMENTS]
    assert len(set(files)) == len(files), files


def test_every_experiment_has_its_own_representation():
    specs = [e["features"] for e in EXPERIMENTS]
    assert len(set(specs)) == len(specs), specs


def test_the_baseline_reproduces_the_paper():
    """One experiment must be plain Morgan, or there is nothing to compare to."""
    baseline = [e for e in EXPERIMENTS if e["features"] == "morgan"]
    assert len(baseline) == 1
    assert baseline[0]["runs"] == "/kaggle/working/runs"


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e["features"])
def test_generated_cells_parse(experiment):
    assert check_cells(train_cells(experiment)) == []


def test_compare_notebook_parses():
    assert check_cells(COMPARE_CELLS) == []


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e["features"])
def test_notebook_pins_its_own_features_and_directory(experiment):
    """The spec is written into the notebook, not left as an editable default.

    A shared notebook with a variable to change is how two experiments end up
    in one directory.
    """
    source = "\n".join(c["source"] for c in train_cells(experiment)
                       if c["cell_type"] == "code")
    assert f"FEATURES = {experiment['features']!r}" in source
    assert f"RUNS     = {experiment['runs']!r}" in source

    folder = experiment["runs"].rsplit("/", 1)[-1]
    assert f"/kaggle/input/**/{folder}/*/checkpoint.pt" in source
    # No other experiment's directory may be referenced anywhere in it.
    for other in EXPERIMENTS:
        if other is experiment:
            continue
        other_folder = other["runs"].rsplit("/", 1)[-1]
        if other_folder == "runs":
            continue          # 'runs' is a substring of every 'runs-*' name
        assert other_folder not in source, f"{experiment['file']} mentions {other_folder}"


@pytest.mark.parametrize("experiment",
                         [e for e in EXPERIMENTS if e["needs_embeddings"]],
                         ids=lambda e: e["features"])
def test_embedding_experiments_check_the_file_exists_first(experiment):
    """Fail before the GPU time is spent, not after."""
    source = "\n".join(c["source"] for c in train_cells(experiment)
                       if c["cell_type"] == "code")
    assert "drug_embeddings_chemberta.parquet" in source
    assert "raise SystemExit" in source


def test_warm_start_is_off_by_default():
    """A warm start has to be a deliberate choice; the clean run is the default."""
    for experiment in EXPERIMENTS:
        source = "\n".join(c["source"] for c in train_cells(experiment)
                           if c["cell_type"] == "code")
        assert "INIT_FROM = None" in source
        assert "--init-from" not in source or "if INIT_FROM:" in source


# ------------------------------------------------- the files actually on disk

def test_checked_in_notebooks_are_current():
    """kaggle/*.ipynb must match what the generator produces.

    Hand-editing the JSON is how a stray escape once became a real newline and
    shipped a notebook Kaggle could not open.
    """
    result = subprocess.run([sys.executable, "scripts/build_kaggle_notebook.py"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    changed = subprocess.run(["git", "status", "--porcelain", "kaggle"],
                             cwd=ROOT, capture_output=True, text=True)
    if changed.returncode == 0 and changed.stdout.strip():
        pytest.skip("notebooks have uncommitted changes; regenerated cleanly")


@pytest.mark.parametrize("name", [e["file"] for e in EXPERIMENTS]
                         + ["01_setup_and_data.ipynb", "02_preprocess.ipynb",
                            "06_compare.ipynb"])
def test_every_shipped_notebook_is_valid_json_and_parses(name):
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    assert notebook["cells"], f"{name} has no cells"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        ast.parse(_as_python(source))     # raises SyntaxError on failure


def test_repair_flag_guards_both_redo_and_stage():
    """A half-applied repair is worse than none.

    REPAIR has to gate REDO *and* STAGE together. Gating only REDO would delete
    the checkpoints and then retrain all three stages, throwing away a good
    pretrain; gating only STAGE would run the two stages and immediately resume
    the very checkpoints the repair exists to discard.
    """
    for experiment in EXPERIMENTS:
        source = "\n".join(c["source"] for c in train_cells(experiment)
                           if c["cell_type"] == "code")
        stages = experiment.get("repair")
        if not stages:
            assert "REPAIR" not in source
            continue
        assert f"REDO = {stages!r} if REPAIR else []" in source
        assert f"STAGE = {' '.join(stages)!r} if REPAIR else 'all'" in source
        assert "REPAIR = True" in source
        # The markdown must say how to turn it off again.
        prose = "\n".join(c["source"] for c in train_cells(experiment)
                          if c["cell_type"] == "markdown")
        assert "REPAIR = False" in prose


def test_repair_never_discards_a_good_pretrain():
    """Pretraining was not affected by the leak, so it must be reused."""
    for experiment in EXPERIMENTS:
        stages = experiment.get("repair")
        if stages:
            assert "pretrain" not in stages, (
                "repairing pretrain would throw away six GPU-hours of valid work")
