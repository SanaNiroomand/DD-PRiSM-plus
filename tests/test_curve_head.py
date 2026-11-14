"""Widening the curve head must change the curve head and nothing else.

The published model reads all four Hill-curve parameters off a 2-dimensional
non-negative vector, so they span at most rank 2 and in practice rank 1. These
tests pin three things: that the default still reproduces the paper exactly,
that widening touches only the last layer and the four heads, and that the rank
measurement the report quotes actually measures what it says.

Run with:  python -m pytest tests/test_curve_head.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ddprism.monotherapy import MonotherapyModel as VectorisedMonotherapyModel
from ddprism.pathways import synthetic_gene_set
from ddprism.train import CURVE_HEADS, adapt_curve_head
from measure_curve_rank import effective_rank
from original.ddprism_original import MonotherapyModel


def gene_set():
    return synthetic_gene_set(num_pathways=8, seed=0)


def inputs(model_gene_set, batch=16, drug_dim=512):
    return ([torch.randn(batch, len(g)) for g in model_gene_set.values()],
            torch.randint(0, 2, (batch, drug_dim)).float(),
            torch.randn(batch, 1))


# ------------------------------------------------- the default is the paper

def test_default_width_is_the_published_two():
    """Anything else and this class stops reproducing the paper."""
    model = VectorisedMonotherapyModel(gene_set(), num_buckets=2)
    assert model.curve_width == 2
    assert model.final_y_max.in_features == 2
    assert isinstance(model.final_block[-1], torch.nn.ReLU)


def test_adapting_to_two_is_a_no_op():
    """Passing the published width must not silently re-initialise the head."""
    torch.manual_seed(0)
    model = MonotherapyModel(gene_set())
    before = {name: getattr(model, name).weight.clone() for name in CURVE_HEADS}
    before_block = model.final_block[-3].weight.clone()

    adapt_curve_head(model, 2)

    for name in CURVE_HEADS:
        torch.testing.assert_close(getattr(model, name).weight, before[name])
    torch.testing.assert_close(model.final_block[-3].weight, before_block)


# ------------------------------------------------- widening changes one thing

@pytest.mark.parametrize("width", [4, 8, 16])
def test_only_the_curve_head_changes_shape(width):
    baseline = MonotherapyModel(gene_set())
    widened = adapt_curve_head(MonotherapyModel(gene_set()), width)

    # The last Linear of final_block, its BatchNorm, and the four heads.
    last = max(i for i, m in enumerate(widened.final_block)
               if isinstance(m, torch.nn.Linear))
    changed = {f"{prefix}.{suffix}"
               for prefix in (*CURVE_HEADS, f"final_block.{last}",
                              f"final_block.{last + 1}")
               for suffix in ("weight", "bias")}

    for name, parameter in baseline.named_parameters():
        other = dict(widened.named_parameters())[name]
        if name in changed:
            continue
        assert other.shape == parameter.shape, f"{name} changed shape"

    assert widened.final_y_max.in_features == width
    assert widened.final_block[last].out_features == width


@pytest.mark.parametrize("width", [2, 8])
def test_widened_model_runs_end_to_end(width):
    model_gene_set = gene_set()
    model = adapt_curve_head(MonotherapyModel(model_gene_set), width).eval()
    output = model(list(inputs(model_gene_set)))
    assert output.shape == (16, 1)
    assert torch.isfinite(output).all()


def test_vectorised_model_takes_the_same_width():
    model_gene_set = gene_set()
    model = VectorisedMonotherapyModel(model_gene_set, num_buckets=2,
                                       curve_width=8).eval()
    per_pathway, fingerprint, dose = inputs(model_gene_set)
    output = model(model.pack(per_pathway), fingerprint, dose)
    assert output.shape == (16, 1)


def test_curve_relu_can_be_dropped():
    """The ablation: without the ReLU the parameters may go negative."""
    model_gene_set = gene_set()
    model = VectorisedMonotherapyModel(model_gene_set, num_buckets=2,
                                       curve_width=8, curve_relu=False)
    assert not isinstance(model.final_block[-1], torch.nn.ReLU)
    assert isinstance(model.final_block[-1], torch.nn.BatchNorm1d)


def test_adapting_twice_does_not_corrupt_the_block():
    """Slicing by a fixed module count would break on the second call."""
    model = adapt_curve_head(MonotherapyModel(gene_set()), 8)
    model = adapt_curve_head(model, 16, curve_relu=False)

    linears = [m for m in model.final_block if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in linears] == [32, 8, 16]
    assert model.final_y_max.in_features == 16


def test_widening_adds_only_a_handful_of_parameters():
    """The change must not be confusable with 'we used a bigger network'."""
    baseline = sum(p.numel() for p in MonotherapyModel(gene_set()).parameters())
    widened = sum(p.numel()
                  for p in adapt_curve_head(MonotherapyModel(gene_set()), 8).parameters())
    assert 0 < widened - baseline < 1000, widened - baseline


# ------------------------------------------------- the rank measurement

def test_effective_rank_counts_independent_directions():
    generator = np.random.default_rng(0)
    independent = generator.standard_normal((500, 4))
    assert effective_rank(independent)[0] == 4


def test_effective_rank_sees_through_one_number_in_four_hats():
    """Four affine functions of one scalar are rank 1, however scaled."""
    generator = np.random.default_rng(0)
    scalar = generator.standard_normal((500, 1))
    disguised = np.hstack([3 * scalar + 1, -2 * scalar, 0.5 * scalar, scalar - 7])
    assert effective_rank(disguised)[0] == 1


def test_effective_rank_is_not_fooled_by_an_offset():
    """Centring is why: an uncentred SVD scores a constant column as a rank."""
    generator = np.random.default_rng(0)
    scalar = generator.standard_normal((500, 1))
    with_offset = np.hstack([scalar, scalar + 100, scalar, scalar])
    assert effective_rank(with_offset)[0] == 1


def test_a_width_two_head_cannot_exceed_rank_two():
    """The structural claim, checked rather than asserted.

    Four Linear(2, 1) heads over one 2-dimensional vector span at most two
    directions no matter how they are trained. This is the defect; the measured
    rank of 1 on the authors' model is that bound being hit harder still.
    """
    torch.manual_seed(0)
    model = MonotherapyModel(gene_set()).eval()
    per_pathway, fingerprint, dose = inputs(gene_set(), batch=256)

    captured = {}
    model.final_block.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("embed", out))
    with torch.no_grad():
        model([per_pathway, fingerprint, dose])
        parameters = torch.cat([getattr(model, name)(captured["embed"])
                                for name in CURVE_HEADS], dim=1)

    assert captured["embed"].shape[1] == 2
    assert effective_rank(parameters.numpy())[0] <= 2
