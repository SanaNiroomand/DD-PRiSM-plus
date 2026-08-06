"""Bookkeeping for the 186 variable-size KEGG pathways.

The Monotherapy model gives every pathway its own small sub-network, sized by
how many genes that pathway contains. To evaluate them as batched ops instead
of a Python loop, pathways have to be padded to a common width -- but padding
all 186 to the widest one is a bad trade. The gene-attention layer is
Linear(n_p + d_p, n_p), so its cost grows with n_p squared, and KEGG sizes are
badly skewed (median around 55 genes, maximum near 390). Padding globally
inflates the real work by roughly fifty times.

So we sort pathways by size and pad only within same-size buckets. A handful of
batched ops replaces the 186-step loop while keeping wasted work small.
"""

import math

import torch


class PathwayBucket:
    """One group of similarly-sized pathways, padded to the group's widest."""

    def __init__(self, indices, gene_counts, drug_counts):
        self.indices = torch.tensor(indices, dtype=torch.long)
        self.gene_counts = gene_counts
        self.drug_counts = drug_counts
        self.size = len(indices)
        self.max_genes = max(gene_counts)
        self.max_drug = max(drug_counts)
        self.gene_mask = _mask(gene_counts, self.max_genes)
        self.drug_mask = _mask(drug_counts, self.max_drug)

    @property
    def attention_weights(self):
        return self.size * self.max_genes * (self.max_genes + self.max_drug)


def _mask(counts, width):
    mask = torch.zeros(len(counts), width, dtype=torch.bool)
    for i, n in enumerate(counts):
        mask[i, :n] = True
    return mask


class PathwaySpec:
    """Sizes, buckets, masks and orderings for a set of pathways.

    Args:
        gene_set: mapping of pathway name -> member genes. Insertion order
            defines the canonical pathway index order.
        num_buckets: how many size buckets to split into. 1 reproduces naive
            global padding; the whole point is to use more than that.
    """

    def __init__(self, gene_set, num_buckets=16):
        self.gene_set = gene_set
        self.names = list(gene_set.keys())
        self.num_pathways = len(self.names)

        self.gene_counts = [len(gene_set[name]) for name in self.names]
        # Matches the reference: drug_for_pathway_size = int(input_size/4)+1
        self.drug_counts = [n // 4 + 1 for n in self.gene_counts]

        self.max_genes = max(self.gene_counts)
        self.max_drug = max(self.drug_counts)
        self.num_buckets = max(1, min(num_buckets, self.num_pathways))
        self.buckets = self._build_buckets()

        # Pathways are processed in bucket order; this permutation puts the
        # results back into the caller's original order.
        order = torch.cat([bucket.indices for bucket in self.buckets])
        self.order = order
        self.inverse_order = torch.argsort(order)

    def _build_buckets(self):
        by_size = sorted(range(self.num_pathways), key=lambda i: self.gene_counts[i])
        per_bucket = math.ceil(self.num_pathways / self.num_buckets)
        buckets = []
        for start in range(0, self.num_pathways, per_bucket):
            chunk = by_size[start:start + per_bucket]
            buckets.append(PathwayBucket(
                chunk,
                [self.gene_counts[i] for i in chunk],
                [self.drug_counts[i] for i in chunk],
            ))
        return buckets

    @property
    def real_attention_weights(self):
        return sum(n * (n + d) for n, d in zip(self.gene_counts, self.drug_counts))

    @property
    def padded_attention_weights(self):
        return sum(bucket.attention_weights for bucket in self.buckets)

    @property
    def padding_overhead(self):
        """Padded attention work divided by the irreducible amount."""
        return self.padded_attention_weights / self.real_attention_weights

    def pack(self, per_pathway):
        """Group a list of P tensors, shaped (B, n_p), into per-bucket batches.

        Returns one (B, bucket_size, bucket_max_genes) tensor per bucket.
        """
        batch = per_pathway[0].shape[0]
        packed = []
        for bucket in self.buckets:
            out = per_pathway[0].new_zeros(batch, bucket.size, bucket.max_genes)
            for local, index in enumerate(bucket.indices.tolist()):
                tensor = per_pathway[index]
                out[:, local, : tensor.shape[1]] = tensor
            packed.append(out)
        return packed

    def __len__(self):
        return self.num_pathways


def read_gmt(path):
    """Read an MSigDB .gmt file into {pathway_name: [genes]}."""
    gene_set = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            gene_set[fields[0]] = fields[2:]
    return gene_set


def synthetic_gene_set(num_pathways=186, seed=0):
    """A stand-in gene set with KEGG-legacy-like size distribution.

    Used by the tests so they run without the MSigDB download, which now
    requires a registered login. Sizes are lognormal, clipped to the range
    KEGG legacy actually spans (about 10 to 390 genes).
    """
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randn(num_pathways, generator=generator) * 0.75 + math.log(55)
    sizes = draws.exp().round().clamp(10, 390).to(torch.int64).tolist()
    return {f"KEGG_SYNTHETIC_{i:03d}": [f"G{i}_{j}" for j in range(n)]
            for i, n in enumerate(sizes)}
