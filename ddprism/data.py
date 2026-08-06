"""Device-resident dataset for the Monotherapy model.

The published ``MonotherapyDataset`` rebuilds 186 tensors per batch out of
pandas object columns, which cost about 50 ms per batch of 1024 -- fine when
the model took 560 ms on CPU, but more than half of every step once the model
runs at 44 ms on a T4.

None of that work is necessary. There are only 66 cell lines, so their pathway
expression is a small fixed table that can be packed once and parked on the
GPU. Fingerprints are equally bounded. What is left per batch is an index
gather, which costs effectively nothing and needs no worker processes.

    data = MonotherapyTensorData(spec, expression_by_cellline, fingerprints)
    batches = MonotherapyBatches(data, cell_ids, drug_ids, dose, viability)
    for (genes, fp, dose), y in batches:
        prediction = model(genes, fp, dose)
"""

import numpy as np
import torch


def _index_and_values(table, ids=None):
    """Accept a DataFrame, a dict, or (ids, array) and return (ids, ndarray)."""
    if hasattr(table, "index") and hasattr(table, "values"):
        return list(table.index), np.asarray(table.values)
    if isinstance(table, dict):
        keys = list(table.keys())
        return keys, np.asarray([table[k] for k in keys])
    if ids is None:
        raise ValueError("ids must be given when passing a bare array")
    return list(ids), np.asarray(table)


class MonotherapyTensorData:
    """Pathway expression and fingerprints, packed once and held on device.

    Args:
        spec: the :class:`~ddprism.pathways.PathwaySpec` the model was built
            with. Bucket layout must match or the gathers will be misaligned.
        expression_by_cellline: mapping cell line -> sequence of P arrays, in
            the same pathway order as ``spec.names``.
        fingerprints: DataFrame indexed by drug id, a dict, or an array plus
            ``drug_ids``. Stored as uint8 and cast per batch; 512-bit Morgan
            fingerprints are binary, so this costs nothing and saves 4x memory.
    """

    def __init__(self, spec, expression_by_cellline, fingerprints,
                 drug_ids=None, device="cpu"):
        self.spec = spec
        self.device = torch.device(device)

        self.cell_ids = list(expression_by_cellline.keys())
        self.cell_index = {name: i for i, name in enumerate(self.cell_ids)}

        num_cells = len(self.cell_ids)
        self.expression = []
        for bucket in spec.buckets:
            block = torch.zeros(num_cells, bucket.size, bucket.max_genes,
                                dtype=torch.float32)
            for cell, name in enumerate(self.cell_ids):
                rows = expression_by_cellline[name]
                for local, pathway in enumerate(bucket.indices.tolist()):
                    values = torch.as_tensor(np.asarray(rows[pathway]),
                                             dtype=torch.float32)
                    block[cell, local, : values.numel()] = values
            self.expression.append(block.to(self.device))

        self.drug_ids, fingerprint_array = _index_and_values(fingerprints, drug_ids)
        self.drug_index = {name: i for i, name in enumerate(self.drug_ids)}
        self.fingerprints = torch.as_tensor(
            fingerprint_array, dtype=torch.uint8).to(self.device)

    @property
    def nbytes(self):
        total = sum(block.numel() * block.element_size() for block in self.expression)
        return total + self.fingerprints.numel() * self.fingerprints.element_size()

    def describe(self):
        return (f"{len(self.cell_ids)} cell lines x {len(self.spec)} pathways, "
                f"{len(self.drug_ids):,} drugs, {self.nbytes / 1e6:.0f} MB on "
                f"{self.device}")

    def gather(self, cell_rows, drug_rows):
        genes = [block[cell_rows] for block in self.expression]
        return genes, self.fingerprints[drug_rows].float()


class MonotherapyBatches:
    """Minibatches drawn straight out of device memory -- no DataLoader.

    Args:
        data: a :class:`MonotherapyTensorData`.
        cell_ids, drug_ids: per-response identifiers, looked up in ``data``.
        dose, viability: per-response floats.
    """

    def __init__(self, data, cell_ids, drug_ids, dose, viability,
                 batch_size=1024, shuffle=True, drop_last=False, generator=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        device = data.device
        self.cell_rows = torch.as_tensor(
            [data.cell_index[c] for c in cell_ids], dtype=torch.long).to(device)
        self.drug_rows = torch.as_tensor(
            [data.drug_index[d] for d in drug_ids], dtype=torch.long).to(device)
        self.dose = torch.as_tensor(
            np.asarray(dose), dtype=torch.float32).to(device).view(-1, 1)
        self.viability = torch.as_tensor(
            np.asarray(viability), dtype=torch.float32).to(device).view(-1, 1)

        if not (len(self.cell_rows) == len(self.drug_rows)
                == len(self.dose) == len(self.viability)):
            raise ValueError("cell_ids, drug_ids, dose and viability must align")

    def __len__(self):
        count = len(self.cell_rows)
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        count = len(self.cell_rows)
        if self.shuffle:
            order = torch.randperm(count, generator=self.generator,
                                   device=self.data.device)
        else:
            order = torch.arange(count, device=self.data.device)

        for start in range(0, count, self.batch_size):
            selected = order[start:start + self.batch_size]
            if self.drop_last and len(selected) < self.batch_size:
                break
            genes, fingerprints = self.data.gather(
                self.cell_rows[selected], self.drug_rows[selected])
            yield (genes, fingerprints, self.dose[selected]), self.viability[selected]
