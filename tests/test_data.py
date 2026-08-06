"""Device-resident batching must feed the model exactly what the old path did.

Run with:  python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.data import MonotherapyBatches, MonotherapyTensorData
from ddprism.monotherapy import MonotherapyModel
from ddprism.pathways import synthetic_gene_set


def build(num_pathways=16, num_cells=8, num_drugs=12, num_rows=40, seed=0):
    rng = np.random.default_rng(seed)
    gene_set = synthetic_gene_set(num_pathways=num_pathways, seed=seed)
    model = MonotherapyModel(gene_set, num_buckets=4)

    cells = [f"CELL_{i}" for i in range(num_cells)]
    drugs = [f"NSC{i}" for i in range(num_drugs)]

    expression = {
        cell: [rng.standard_normal(len(genes)).astype(np.float32)
               for genes in gene_set.values()]
        for cell in cells
    }
    fingerprints = {d: rng.integers(0, 2, 512).astype(np.uint8) for d in drugs}

    data = MonotherapyTensorData(model.spec, expression, fingerprints)
    rows = {
        "cell_ids": [cells[i] for i in rng.integers(0, num_cells, num_rows)],
        "drug_ids": [drugs[i] for i in rng.integers(0, num_drugs, num_rows)],
        "dose": rng.standard_normal(num_rows).astype(np.float32),
        "viability": rng.random(num_rows).astype(np.float32),
    }
    return model, data, rows, expression, fingerprints, gene_set


def test_gathered_genes_match_manual_packing():
    """A gathered batch must equal spec.pack() of the same cell lines."""
    model, data, rows, expression, _, gene_set = build()
    batches = MonotherapyBatches(data, batch_size=7, shuffle=False, **rows)

    (genes, _, _), _ = next(iter(batches))
    wanted = rows["cell_ids"][:7]

    per_pathway = [
        torch.tensor(np.stack([expression[c][p] for c in wanted]))
        for p in range(len(gene_set))
    ]
    expected = model.pack(per_pathway)

    assert len(genes) == len(expected)
    for actual, want in zip(genes, expected):
        torch.testing.assert_close(actual, want)


def test_model_output_identical_to_manual_path():
    model, data, rows, expression, fingerprints, gene_set = build()
    model.eval()
    batches = MonotherapyBatches(data, batch_size=40, shuffle=False, **rows)

    (genes, fp, dose), y = next(iter(batches))
    fast = model(genes, fp, dose)

    per_pathway = [
        torch.tensor(np.stack([expression[c][p] for c in rows["cell_ids"]]))
        for p in range(len(gene_set))
    ]
    manual_fp = torch.tensor(
        np.stack([fingerprints[d] for d in rows["drug_ids"]]), dtype=torch.float32)
    manual_dose = torch.tensor(rows["dose"]).view(-1, 1)
    manual = model(model.pack(per_pathway), manual_fp, manual_dose)

    torch.testing.assert_close(fast, manual)
    torch.testing.assert_close(y.view(-1), torch.tensor(rows["viability"]))


def test_epoch_visits_every_row_exactly_once():
    _, data, rows, _, _, _ = build(num_rows=40)
    batches = MonotherapyBatches(data, batch_size=7, shuffle=True, **rows)

    seen = torch.cat([y.view(-1) for _, y in batches])
    assert len(seen) == 40
    torch.testing.assert_close(
        seen.sort().values,
        torch.tensor(rows["viability"]).sort().values)


def test_shuffle_changes_order_but_not_content():
    _, data, rows, _, _, _ = build(num_rows=40)
    ordered = MonotherapyBatches(data, batch_size=40, shuffle=False, **rows)
    shuffled = MonotherapyBatches(
        data, batch_size=40, shuffle=True,
        generator=torch.Generator().manual_seed(7), **rows)

    a = torch.cat([y.view(-1) for _, y in ordered])
    b = torch.cat([y.view(-1) for _, y in shuffled])

    assert not torch.equal(a, b)
    torch.testing.assert_close(a.sort().values, b.sort().values)


def test_len_matches_batches_produced():
    _, data, rows, _, _, _ = build(num_rows=40)
    for batch_size, drop_last in [(7, False), (7, True), (40, False), (64, False)]:
        batches = MonotherapyBatches(data, batch_size=batch_size,
                                     drop_last=drop_last, **rows)
        assert len(batches) == sum(1 for _ in batches)


def test_batch_shapes():
    model, data, rows, _, _, _ = build()
    batches = MonotherapyBatches(data, batch_size=7, shuffle=False, **rows)
    (genes, fp, dose), y = next(iter(batches))

    assert fp.shape == (7, 512) and fp.dtype == torch.float32
    assert dose.shape == (7, 1) and y.shape == (7, 1)
    for tensor, bucket in zip(genes, model.spec.buckets):
        assert tensor.shape == (7, bucket.size, bucket.max_genes)
