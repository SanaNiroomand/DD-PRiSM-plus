"""Swapping the drug representation must change the drug input and nothing else.

The experiment only means something if the comparison is clean: if widening the
drug features also widened the rest of the network, a better score would say
"more parameters help", not "better chemistry helps". These tests pin that down,
along with the two things that silently ruin a fusion run -- mismatched scales
between the two blocks, and drugs that quietly lose their feature vector.

Run with:  python -m pytest tests/test_drugfeatures.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddprism.drugfeatures import (load_drug_features, parse_spec, standardise,
                                  used_drug_ids)
from ddprism.monotherapy import MonotherapyModel as VectorisedMonotherapyModel
from ddprism.pathways import synthetic_gene_set
from ddprism.train import adapt_drug_input, warm_start
from original.ddprism_original import MonotherapyModel


@pytest.fixture
def processed(tmp_path):
    """A drug library with Morgan bits and a fake pretrained embedding."""
    rng = np.random.default_rng(0)
    root = tmp_path / "processed"
    root.mkdir()

    nsc = list(range(100, 140))
    morgan = pd.DataFrame(rng.integers(0, 2, (len(nsc), 512)).astype(np.uint8),
                          index=pd.Index(nsc, name="NSC"))
    morgan.columns = [str(c) for c in morgan.columns]
    morgan.to_parquet(root / "fingerprints.parquet")

    # Deliberately off-scale: mean 40, std 7. Raw, this block would swamp the
    # 0/1 fingerprint sitting next to it.
    embedding = pd.DataFrame(
        (rng.standard_normal((len(nsc), 384)) * 7 + 40).astype(np.float32),
        index=pd.Index(nsc, name="NSC"))
    embedding.columns = [f"e{i}" for i in range(384)]
    # One dead dimension, to check it is not divided by ~zero.
    embedding["e0"] = 1.0
    embedding.to_parquet(root / "drug_embeddings_chemberta.parquet")

    responses = pd.DataFrame({"NSC": rng.choice(nsc[:30], 200)})
    responses.to_parquet(root / "nci60_filtered.parquet")
    return root


# ------------------------------------------------------------------ assembly

def test_fusion_concatenates_both_blocks(processed):
    _, matrix, report = load_drug_features(processed, "morgan+chemberta")
    assert matrix.shape == (40, 512 + 384)
    assert report["width"] == 896
    assert [s["name"] for s in report["sources"]] == ["morgan", "chemberta"]


def test_morgan_alone_stays_uint8(processed):
    """The paper's path must not start paying 4x memory for nothing."""
    _, matrix, _ = load_drug_features(processed, "morgan")
    assert matrix.dtype == np.uint8
    assert matrix.shape == (40, 512)


def test_embeddings_are_standardised_before_fusion(processed):
    """Raw, the embedding block would dominate the fingerprint beside it."""
    _, fused, _ = load_drug_features(processed, "morgan+chemberta")
    morgan, embedding = fused[:, :512], fused[:, 512:]

    assert abs(embedding.mean()) < 1e-5
    assert abs(embedding[:, 1:].std() - 1.0) < 0.05
    # The fingerprint half is untouched: still exactly 0/1.
    assert set(np.unique(morgan)) <= {0.0, 1.0}


def test_constant_dimension_survives_standardisation(processed):
    _, fused, _ = load_drug_features(processed, "morgan+chemberta")
    dead = fused[:, 512]                       # e0, constant at 1.0
    assert np.isfinite(dead).all()
    assert np.allclose(dead, 0.0)


def test_standardize_can_be_switched_off(processed):
    _, raw, _ = load_drug_features(processed, "chemberta", standardize=False)
    assert raw.mean() > 30            # native scale preserved


def test_restrict_keeps_only_the_drugs_asked_for(processed):
    ids, matrix, report = load_drug_features(processed, "morgan",
                                             restrict=[100, 101, 102])
    assert list(ids) == [100, 101, 102]
    assert matrix.shape[0] == 3
    assert report["unavailable"] == 0


def test_missing_feature_vectors_are_reported_not_hidden(processed):
    """A drug with no vector must be counted, not silently dropped."""
    _, _, report = load_drug_features(processed, "morgan",
                                      restrict=[100, 101, 999_999])
    assert report["requested"] == 3
    assert report["unavailable"] == 1
    assert report["drugs"] == 2


def test_sources_are_intersected_on_nsc(processed):
    """A drug present in one source and not the other cannot be used."""
    extra = pd.read_parquet(processed / "fingerprints.parquet")
    extra.loc[9999] = 0
    extra.to_parquet(processed / "fingerprints.parquet")

    _, morgan_only, _ = load_drug_features(processed, "morgan")
    _, fused, _ = load_drug_features(processed, "morgan+chemberta")
    assert morgan_only.shape[0] == 41
    assert fused.shape[0] == 40


def test_missing_embedding_file_says_how_to_build_it(tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    pd.DataFrame(np.zeros((2, 512), np.uint8),
                 index=pd.Index([1, 2], name="NSC")).to_parquet(
        root / "fingerprints.parquet")

    with pytest.raises(SystemExit, match="embed_drugs.py"):
        load_drug_features(root, "morgan+chemberta")


def test_used_drug_ids_reads_the_response_tables(processed):
    ids = used_drug_ids(processed)
    assert ids is not None and ids <= set(range(100, 130))


@pytest.mark.parametrize("spec,expected", [
    ("morgan", ["morgan"]),
    ("morgan+chemberta", ["morgan", "chemberta"]),
    ("morgan,molformer", ["morgan", "molformer"]),
    (["chemberta"], ["chemberta"]),
])
def test_spec_parsing(spec, expected):
    assert parse_spec(spec) == expected


def test_unknown_source_is_rejected():
    with pytest.raises(SystemExit, match="unknown drug feature source"):
        parse_spec("morgan+gpt4")


def test_standardise_is_exact():
    values = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    out = standardise(values)
    assert np.allclose(out[:, 0], [-1.2247449, 0.0, 1.2247449], atol=1e-5)
    assert np.allclose(out[:, 1], 0.0)


# --------------------------------------------------------------- model shape

def _gene_set():
    return synthetic_gene_set(num_pathways=8, seed=0)


@pytest.mark.parametrize("drug_dim", [512, 896, 1280])
def test_only_the_drug_input_layer_changes_width(drug_dim):
    """Everything after the first Linear keeps its published shape."""
    baseline = MonotherapyModel(_gene_set())
    widened = adapt_drug_input(MonotherapyModel(_gene_set()), drug_dim)

    assert widened.drug_block[0].in_features == drug_dim
    assert widened.new_drug_block[0].in_features == drug_dim

    for name, parameter in baseline.named_parameters():
        other = dict(widened.named_parameters())[name]
        if name in ("drug_block.0.weight", "new_drug_block.0.weight"):
            assert other.shape[1] == drug_dim
            assert other.shape[0] == parameter.shape[0]
        else:
            assert other.shape == parameter.shape, f"{name} changed shape"


def test_adapting_to_the_same_width_is_a_no_op():
    """Passing 512 must not silently re-initialise the paper's own layers."""
    model = MonotherapyModel(_gene_set())
    before = model.drug_block[0].weight.clone()
    adapt_drug_input(model, 512)
    torch.testing.assert_close(model.drug_block[0].weight, before)


def test_vectorised_model_accepts_the_same_widths():
    gene_set = _gene_set()
    model = VectorisedMonotherapyModel(gene_set, num_buckets=2, drug_dim=896)
    per_pathway = [torch.randn(4, len(g)) for g in gene_set.values()]
    output = model.eval()(model.pack(per_pathway), torch.randn(4, 896),
                          torch.randn(4, 1))
    assert output.shape == (4, 1)


def test_widened_model_still_runs_end_to_end():
    gene_set = _gene_set()
    model = adapt_drug_input(MonotherapyModel(gene_set), 896).eval()
    per_pathway = [torch.randn(4, len(g)) for g in gene_set.values()]
    output = model([per_pathway, torch.randn(4, 896), torch.randn(4, 1)])
    assert output.shape == (4, 1)
    assert torch.isfinite(output).all()


# ---------------------------------------------------------------- warm start

def test_warm_start_reproduces_the_narrow_model_exactly(tmp_path):
    """At step zero the widened model must compute what the old one computed.

    The new columns start at zero, so the extra features contribute nothing
    until training moves them. If this drifts, a warm-started run is not
    comparable with the checkpoint it started from.
    """
    torch.manual_seed(0)
    gene_set = _gene_set()
    narrow = MonotherapyModel(gene_set)

    # Give BatchNorm real running statistics, so eval mode is exercised.
    narrow.train()
    for _ in range(3):
        narrow([[torch.randn(8, len(g)) for g in gene_set.values()],
                torch.randint(0, 2, (8, 512)).float(), torch.randn(8, 1)])
    torch.save({"model": narrow.state_dict()}, tmp_path / "best.pt")

    wide = adapt_drug_input(MonotherapyModel(gene_set), 896)
    warm_start(wide, tmp_path / "best.pt", torch.device("cpu"))

    narrow.eval()
    wide.eval()
    per_pathway = [torch.randn(6, len(g)) for g in gene_set.values()]
    fingerprint = torch.randint(0, 2, (6, 512)).float()
    embedding = torch.randn(6, 384) * 3        # arbitrary; must be ignored
    dose = torch.randn(6, 1)

    expected = narrow([per_pathway, fingerprint, dose])
    actual = wide([per_pathway, torch.cat([fingerprint, embedding], dim=1), dose])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_warm_start_leaves_the_new_columns_at_zero(tmp_path):
    torch.manual_seed(0)
    gene_set = _gene_set()
    torch.save({"model": MonotherapyModel(gene_set).state_dict()},
               tmp_path / "best.pt")

    wide = adapt_drug_input(MonotherapyModel(gene_set), 896)
    warm_start(wide, tmp_path / "best.pt", torch.device("cpu"))

    assert wide.drug_block[0].weight[:, 512:].abs().max() == 0
    assert wide.new_drug_block[0].weight[:, 512:].abs().max() == 0
    assert wide.drug_block[0].weight[:, :512].abs().max() > 0


def test_warm_start_lets_gradients_reach_the_new_columns(tmp_path):
    """Zero-initialised is not the same as dead -- the input is not zero."""
    torch.manual_seed(0)
    gene_set = _gene_set()
    torch.save({"model": MonotherapyModel(gene_set).state_dict()},
               tmp_path / "best.pt")
    wide = adapt_drug_input(MonotherapyModel(gene_set), 896)
    warm_start(wide, tmp_path / "best.pt", torch.device("cpu"))

    wide.train()
    output = wide([[torch.randn(8, len(g)) for g in gene_set.values()],
                   torch.randn(8, 896), torch.randn(8, 1)])
    output.sum().backward()

    assert wide.drug_block[0].weight.grad[:, 512:].abs().sum() > 0
