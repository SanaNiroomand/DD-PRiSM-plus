"""The vectorised model must match the published one, weight for weight.

Run with:  python -m pytest tests -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.monotherapy import MonotherapyModel
from ddprism.pathways import synthetic_gene_set
from original.ddprism_original import MonotherapyModel as ReferenceMonotherapyModel

BUCKET_COUNTS = [1, 4, 16, 24]


def make_batch(gene_set, batch_size=16, dtype=torch.float64):
    per_pathway = [torch.randn(batch_size, len(genes), dtype=dtype)
                   for genes in gene_set.values()]
    drug_fp = torch.randint(0, 2, (batch_size, 512)).to(dtype)
    dose = torch.randn(batch_size, 1, dtype=dtype)
    return [per_pathway, drug_fp, dose]


def build_pair(num_pathways=24, num_buckets=4, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    gene_set = synthetic_gene_set(num_pathways=num_pathways, seed=seed)

    reference = ReferenceMonotherapyModel(gene_set).to(dtype)
    fast = MonotherapyModel(gene_set, num_buckets=num_buckets).to(dtype)

    # Give the reference non-trivial BatchNorm running stats, so eval mode is
    # actually exercised rather than comparing two fresh initialisations.
    reference.train()
    for _ in range(3):
        reference(make_batch(gene_set, batch_size=16, dtype=dtype))

    fast.load_from_reference(reference)
    return reference, fast, gene_set


@pytest.mark.parametrize("num_buckets", BUCKET_COUNTS)
@pytest.mark.parametrize("training", [False, True])
def test_viability_matches_reference(num_buckets, training):
    reference, fast, gene_set = build_pair(num_buckets=num_buckets)
    per_pathway, drug_fp, dose = make_batch(gene_set, batch_size=32)

    reference.train(training)
    fast.train(training)

    expected = reference([per_pathway, drug_fp, dose])
    actual = fast(fast.pack(per_pathway), drug_fp, dose)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("num_buckets", BUCKET_COUNTS)
def test_pathway_attention_matches_reference(num_buckets):
    """The pathway attention the Combination model consumes must also match.

    The authors' forward returns viability only; they recover the attention by
    registering a forward hook on ``sample_attention_block`` and reading its
    output. That hook is what this compares against -- our ``return_attention``
    is a convenience, and it has to agree with their mechanism exactly.
    """
    from original.ddprism_original import Hook

    reference, fast, gene_set = build_pair(num_buckets=num_buckets)
    per_pathway, drug_fp, dose = make_batch(gene_set, batch_size=32)

    reference.eval()
    fast.eval()

    hook = Hook(reference.sample_attention_block)
    reference([per_pathway, drug_fp, dose])
    expected = hook.o
    hook.close()

    _, actual = fast(fast.pack(per_pathway), drug_fp, dose, return_attention=True)

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    # Attention is a softmax over pathways, so rows sum to one.
    torch.testing.assert_close(
        actual.sum(dim=1), torch.ones(actual.shape[0], dtype=actual.dtype))


def test_bucketing_does_not_change_results():
    """Every bucket count must agree with every other, not just the reference."""
    outputs = []
    for num_buckets in BUCKET_COUNTS:
        _, fast, gene_set = build_pair(num_buckets=num_buckets)
        torch.manual_seed(99)
        per_pathway, drug_fp, dose = make_batch(gene_set, batch_size=8)
        fast.eval()
        outputs.append(fast(fast.pack(per_pathway), drug_fp, dose))

    for other in outputs[1:]:
        torch.testing.assert_close(other, outputs[0], rtol=1e-9, atol=1e-9)


def test_float32_matches_within_tolerance():
    """Same check in the dtype training actually uses."""
    reference, fast, gene_set = build_pair(dtype=torch.float32)
    per_pathway, drug_fp, dose = make_batch(gene_set, batch_size=32,
                                            dtype=torch.float32)
    reference.eval()
    fast.eval()

    expected = reference([per_pathway, drug_fp, dose])
    actual = fast(fast.pack(per_pathway), drug_fp, dose)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_batchnorm_running_stats_track_reference():
    """Training-mode updates must move both models' BN buffers identically."""
    reference, fast, gene_set = build_pair(num_buckets=4)
    reference.train()
    fast.train()

    for _ in range(4):
        per_pathway, drug_fp, dose = make_batch(gene_set, batch_size=16)
        reference([per_pathway, drug_fp, dose])
        fast(fast.pack(per_pathway), drug_fp, dose)

    for bucket, block in zip(fast.spec.buckets, fast.pathway_blocks):
        for local, index in enumerate(bucket.indices.tolist()):
            name = fast.spec.names[index]
            n_p = fast.spec.gene_counts[index]
            expected = reference.gene_attention_blocks[name][1]
            torch.testing.assert_close(
                block.attention_norm.running_mean[local, :n_p],
                expected.running_mean, rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(
                block.attention_norm.running_var[local, :n_p],
                expected.running_var, rtol=1e-9, atol=1e-9)


def test_padded_genes_do_not_leak():
    """Writing junk into padded slots must not change any output."""
    _, fast, gene_set = build_pair(num_buckets=4)
    fast.eval()
    per_pathway, drug_fp, dose = make_batch(gene_set, batch_size=8)

    clean = fast.pack(per_pathway)
    dirty = []
    for tensor, block in zip(clean, fast.pathway_blocks):
        corrupted = tensor.clone()
        pad = ~block.gene_mask.unsqueeze(0).expand_as(corrupted)
        corrupted[pad] = 1e6
        dirty.append(corrupted)

    torch.testing.assert_close(
        fast(dirty, drug_fp, dose), fast(clean, drug_fp, dose),
        rtol=1e-9, atol=1e-9)


def test_gradients_flow_to_pathway_parameters():
    gene_set = synthetic_gene_set(num_pathways=8, seed=1)
    fast = MonotherapyModel(gene_set, num_buckets=2).to(torch.float64)
    per_pathway, drug_fp, dose = make_batch(gene_set, batch_size=8)

    fast(fast.pack(per_pathway), drug_fp, dose).sum().backward()

    for block in fast.pathway_blocks:
        assert block.attention_linear.weight.grad.abs().sum() > 0
        assert block.drug_linear.weight.grad.abs().sum() > 0
