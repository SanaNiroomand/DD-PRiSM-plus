"""The combination model must obey the identity the paper publishes.

Supplementary Data 3 gives all five quantities for 2,556 real combinations, and
E = alpha*E1 + beta*E2 + gamma holds there to 1.5e-07. These tests assert the
same identity on our implementation, so any refactor that breaks the
decomposition fails here rather than silently producing plausible numbers.

Run with:  python -m pytest tests -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.combination import CombinationTherapyModel
from ddprism.losses import CustomLoss, estimate_density, pearson


def make_batch(batch=16, pathways=186, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    attention1 = torch.softmax(torch.randn(batch, pathways, generator=generator,
                                           dtype=dtype), dim=1)
    attention2 = torch.softmax(torch.randn(batch, pathways, generator=generator,
                                           dtype=dtype), dim=1)
    viability1 = torch.rand(batch, 1, generator=generator, dtype=dtype)
    viability2 = torch.rand(batch, 1, generator=generator, dtype=dtype)
    return attention1, attention2, viability1, viability2


def test_decomposition_identity_holds():
    """E = alpha*E1 + beta*E2 + gamma, exactly as in Supplementary Data 3."""
    model = CombinationTherapyModel().double().eval()
    a1, a2, v1, v2 = make_batch()

    coefficients, synergy, viability = model(a1, a2, v1, v2)

    efficacy = 1 - viability
    rebuilt = (coefficients[:, :1] * (1 - v1)
               + coefficients[:, 1:] * (1 - v2)
               + synergy)
    torch.testing.assert_close(efficacy, rebuilt, rtol=1e-10, atol=1e-10)


def test_coefficients_form_a_softmax_pair():
    model = CombinationTherapyModel().double().eval()
    coefficients, _, _ = model(*make_batch())

    torch.testing.assert_close(coefficients.sum(dim=1),
                               torch.ones(coefficients.shape[0], dtype=torch.float64))
    assert (coefficients >= 0).all()


def test_shapes():
    model = CombinationTherapyModel().double().eval()
    coefficients, synergy, viability = model(*make_batch(batch=7))
    assert coefficients.shape == (7, 2)
    assert synergy.shape == (7, 1)
    assert viability.shape == (7, 1)


def test_unbounded_by_default_matches_the_paper():
    """The published model can emit negative viability; 22% of Data 3 does."""
    model = CombinationTherapyModel(bound_output=False).double().eval()
    with torch.no_grad():
        for block in (model.synergy_network,):
            block[-1].bias.fill_(5.0)      # force a large synergy term
    _, _, viability = model(*make_batch())
    assert (viability < 0).any(), "expected the unbounded model to go negative"


def test_bound_output_prevents_impossible_viability():
    model = CombinationTherapyModel(bound_output=True).double().eval()
    with torch.no_grad():
        model.synergy_network[-1].bias.fill_(5.0)
    _, _, viability = model(*make_batch())
    assert (viability >= 0).all()


def test_gradients_reach_every_subnetwork():
    model = CombinationTherapyModel().double()
    _, _, viability = model(*make_batch())
    viability.sum().backward()

    for name in ("embedding_network", "monotherapy_network", "synergy_network"):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), f"no gradient in {name}"


# ---------------------------------------------------------------- losses

def test_density_weighting_down_weights_the_crowded_region():
    """Labels pile up near 1.0, so that region must be weighted least."""
    torch.manual_seed(0)
    crowded = torch.cat([torch.full((900,), 0.95), torch.rand(100)])
    lut = estimate_density(crowded.numpy())

    loss_fn = CustomLoss(lut, gamma=0.75)
    from ddprism.losses import density_weights
    weight_crowded = density_weights(torch.tensor([[0.95]]), lut).item()
    weight_sparse = density_weights(torch.tensor([[0.20]]), lut).item()

    assert weight_crowded < weight_sparse
    assert 0.25 <= weight_crowded <= 1.0 and weight_sparse <= 1.0


def test_loss_is_zero_only_when_prediction_is_perfect():
    torch.manual_seed(0)
    labels = torch.rand(256, 1)
    lut = estimate_density(labels.numpy())
    loss_fn = CustomLoss(lut)

    exact, mse_exact, corr_exact = loss_fn(labels, labels)
    worse, mse_worse, _ = loss_fn(labels + 0.3, labels)

    assert mse_exact.item() == pytest.approx(0.0, abs=1e-12)
    assert corr_exact.item() == pytest.approx(1.0, abs=1e-6)
    assert exact.item() < worse.item()
    assert mse_worse.item() > mse_exact.item()


def test_pearson_matches_a_known_value():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert pearson(x, x).item() == pytest.approx(1.0, abs=1e-6)
    assert pearson(x, -x).item() == pytest.approx(-1.0, abs=1e-6)
