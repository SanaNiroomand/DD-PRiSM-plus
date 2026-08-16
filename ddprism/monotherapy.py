"""Vectorised Monotherapy model.

Numerically the same network as :mod:`ddprism.reference`, but the 186-step
Python loop over pathways collapses into one batched op per size bucket. It
also returns the pathway attention directly, so the Combination model no longer
needs a forward hook to get at it.
"""

import torch
import torch.nn as nn

from .layers import GroupedBatchNorm1d, GroupedLinear
from .pathways import PathwaySpec


class PathwayBlock(nn.Module):
    """The per-pathway sub-network, evaluated for a whole bucket at once."""

    def __init__(self, bucket):
        super().__init__()
        self.drug_linear = GroupedLinear(
            bucket.size, 32, bucket.max_drug,
            in_counts=[32] * bucket.size, out_counts=bucket.drug_counts)
        self.drug_norm = GroupedBatchNorm1d(
            bucket.size, bucket.max_drug, mask=bucket.drug_mask)

        # Padded layout puts genes in [0, max_genes) and the drug embedding in
        # [max_genes, max_genes + max_drug), so a reference weight matrix has
        # to be scattered into two column blocks rather than copied whole.
        self.attention_linear = GroupedLinear(
            bucket.size, bucket.max_genes + bucket.max_drug, bucket.max_genes,
            in_counts=[n + d for n, d in zip(bucket.gene_counts, bucket.drug_counts)],
            out_counts=bucket.gene_counts)
        self.attention_norm = GroupedBatchNorm1d(
            bucket.size, bucket.max_genes, mask=bucket.gene_mask)

        self.register_buffer("gene_mask", bucket.gene_mask)

    def forward(self, gene_expression, new_drug_embed):
        drug_gene_set = self.drug_norm(self.drug_linear.forward_shared(new_drug_embed))
        # The norm already zeroed padded slots, and relu(0) == 0, so padding
        # contributes nothing to the concat below.
        drug_gene_set = torch.relu(drug_gene_set)

        attention_in = torch.cat([gene_expression, drug_gene_set], dim=2)
        attention = torch.tanh(self.attention_norm(self.attention_linear(attention_in)))
        # Padded genes must not take probability mass from real ones.
        attention = attention.masked_fill(~self.gene_mask, float("-inf"))
        attention = torch.softmax(attention, dim=2)

        return (gene_expression * attention).sum(dim=2)


class MonotherapyModel(nn.Module):
    """Predicts cell viability from pathway expression, a drug, and a dose.

    Args:
        gene_set: mapping of pathway name -> member genes, the same object the
            reference implementation takes.
        num_buckets: pathways are sorted by size and split into this many
            buckets, each padded to its own widest member.
        drug_dim: width of the drug feature vector. 512 is the paper's Morgan
            fingerprint; a larger value comes from concatenating a pretrained
            chemical language model embedding. Only the first Linear of each
            drug branch changes -- everything downstream is untouched, so the
            comparison isolates the representation.

    Forward takes ``gene_expression`` as the per-bucket list produced by
    :meth:`pack`.
    """

    def __init__(self, gene_set, num_buckets=16, drug_dim=512):
        super().__init__()
        self.spec = PathwaySpec(gene_set, num_buckets=num_buckets)
        self.num_pathway = self.spec.num_pathways
        self.drug_dim = drug_dim

        self.drug_block = nn.Sequential(
            nn.Linear(drug_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.new_drug_block = nn.Sequential(
            nn.Linear(drug_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )

        self.pathway_blocks = nn.ModuleList(
            PathwayBlock(bucket) for bucket in self.spec.buckets)
        # P independent BatchNorm1d(1) is exactly BatchNorm1d(P) over (B, P).
        self.pathway_dot_norm = nn.BatchNorm1d(self.num_pathway)

        drug_sample_size = int(self.num_pathway / 16 + 1)
        self.drug_dense_sample_block = nn.Sequential(
            nn.Linear(32, drug_sample_size),
            nn.BatchNorm1d(drug_sample_size),
            nn.ReLU(),
        )
        self.sample_attention_block = nn.Sequential(
            nn.Linear(self.num_pathway + drug_sample_size, self.num_pathway),
            nn.BatchNorm1d(self.num_pathway),
            nn.Tanh(),
            nn.Softmax(dim=1),
        )
        self.concatenated_block = nn.Sequential(
            nn.Linear(128 + self.num_pathway, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.final_block = nn.Sequential(
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 8), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Linear(8, 2), nn.BatchNorm1d(2), nn.ReLU(),
        )

        self.final_y_max = nn.Linear(2, 1)
        self.final_y_min = nn.Linear(2, 1)
        self.final_slope = nn.Linear(2, 1)
        self.final_IC50 = nn.Linear(2, 1)

        self.register_buffer("inverse_order", self.spec.inverse_order)

    def pack(self, per_pathway):
        return self.spec.pack(per_pathway)

    def forward(self, gene_expression, drug_fp, dose, return_attention=False):
        drug_embed = self.drug_block(drug_fp)
        new_drug_embed = self.new_drug_block(drug_fp)

        gene_dot = torch.cat(
            [block(gene_expression[i], new_drug_embed)
             for i, block in enumerate(self.pathway_blocks)], dim=1)
        # Buckets are size-sorted, so restore the caller's pathway order before
        # anything downstream indexes by pathway.
        gene_dot = gene_dot[:, self.inverse_order]
        gene_dot = torch.relu(self.pathway_dot_norm(gene_dot))

        drug_dense_embed = self.drug_dense_sample_block(new_drug_embed)
        drug_effected_concat = torch.cat([gene_dot, drug_dense_embed], dim=1)

        sample_attention = self.sample_attention_block(drug_effected_concat)
        sample_multiplied = gene_dot * sample_attention

        total_concat = torch.cat([sample_multiplied, drug_embed], dim=1)
        final_embed = self.final_block(self.concatenated_block(total_concat))

        y_max = self.final_y_max(final_embed)
        y_min = self.final_y_min(final_embed)
        slope = self.final_slope(final_embed)
        ic50 = self.final_IC50(final_embed)

        sigmoid = torch.sigmoid(-slope * (dose - ic50))
        viability = (y_max - y_min) * sigmoid + y_min

        if return_attention:
            return viability, sample_attention
        return viability

    @torch.no_grad()
    def load_from_reference(self, ref):
        """Copy weights out of a :class:`ReferenceMonotherapyModel`.

        Used by the equivalence test: with identical weights the two models
        must produce identical outputs.
        """
        for name in ("drug_block", "new_drug_block", "drug_dense_sample_block",
                     "sample_attention_block", "concatenated_block", "final_block",
                     "final_y_max", "final_y_min", "final_slope", "final_IC50"):
            getattr(self, name).load_state_dict(getattr(ref, name).state_dict())

        for bucket, block in zip(self.spec.buckets, self.pathway_blocks):
            block.drug_linear.weight.zero_()
            block.drug_linear.bias.zero_()
            block.attention_linear.weight.zero_()
            block.attention_linear.bias.zero_()

            for local, index in enumerate(bucket.indices.tolist()):
                name = self.spec.names[index]
                n_p = self.spec.gene_counts[index]
                d_p = self.spec.drug_counts[index]
                width = bucket.max_genes

                linear = ref.drug_gene_set_blocks[name][0]
                norm = ref.drug_gene_set_blocks[name][1]
                block.drug_linear.weight[local, :d_p, :] = linear.weight
                block.drug_linear.bias[local, :d_p] = linear.bias
                block.drug_norm.weight[local, :d_p] = norm.weight
                block.drug_norm.bias[local, :d_p] = norm.bias
                block.drug_norm.running_mean[local, :d_p] = norm.running_mean
                block.drug_norm.running_var[local, :d_p] = norm.running_var

                linear = ref.gene_attention_blocks[name][0]
                norm = ref.gene_attention_blocks[name][1]
                block.attention_linear.weight[local, :n_p, :n_p] = linear.weight[:, :n_p]
                block.attention_linear.weight[local, :n_p, width:width + d_p] = linear.weight[:, n_p:]
                block.attention_linear.bias[local, :n_p] = linear.bias
                block.attention_norm.weight[local, :n_p] = norm.weight
                block.attention_norm.bias[local, :n_p] = norm.bias
                block.attention_norm.running_mean[local, :n_p] = norm.running_mean
                block.attention_norm.running_var[local, :n_p] = norm.running_var

                norm = ref.gene_dot_blocks[name][0]
                self.pathway_dot_norm.weight[index] = norm.weight[0]
                self.pathway_dot_norm.bias[index] = norm.bias[0]
                self.pathway_dot_norm.running_mean[index] = norm.running_mean[0]
                self.pathway_dot_norm.running_var[index] = norm.running_var[0]

        return self
