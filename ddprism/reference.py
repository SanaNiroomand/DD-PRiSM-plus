"""Verbatim port of the published Monotherapy model.

This mirrors ``MonotherapyModel`` from 00_MonotherapyUtils.ipynb, including the
Python loop over pathways and the two modules the original defines but never
calls. It exists so the vectorised model can be checked against it directly,
without needing a Jupyter kernel. Do not "improve" anything here -- it is the
baseline that defines correct.
"""

import torch
import torch.nn as nn


def batch_dot(tensor1, tensor2, batch_size=1024):
    return (tensor1[None] * tensor2).sum(dim=-1).view(-1, 1)


class ReferenceMonotherapyModel(nn.Module):
    def __init__(self, GeneSet):
        super().__init__()

        self.GeneSet = GeneSet
        self.pathway_list = list(GeneSet.keys())
        self.num_pathway = len(self.pathway_list)

        self.drug_block = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.new_drug_block = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )

        self.drug_gene_set_blocks = nn.ModuleDict()
        self.gene_attention_blocks = nn.ModuleDict()
        self.gene_dot_blocks = nn.ModuleDict()
        for pathway in self.pathway_list:
            input_size = int(len(self.GeneSet[pathway]))
            drug_for_pathway_size = int(input_size / 4) + 1

            self.drug_gene_set_blocks[pathway] = nn.Sequential(
                nn.Linear(32, drug_for_pathway_size),
                nn.BatchNorm1d(drug_for_pathway_size),
                nn.ReLU(),
            )
            self.gene_attention_blocks[pathway] = nn.Sequential(
                nn.Linear(input_size + drug_for_pathway_size, input_size),
                nn.BatchNorm1d(input_size),
                nn.Tanh(),
                nn.Softmax(dim=1),
            )
            self.gene_dot_blocks[pathway] = nn.Sequential(
                nn.BatchNorm1d(1), nn.ReLU(),
            )

        drug_for_pathway_size = int(self.num_pathway / 16 + 1)
        self.drug_dense_sample_block = nn.Sequential(
            nn.Linear(32, drug_for_pathway_size),
            nn.BatchNorm1d(drug_for_pathway_size),
            nn.ReLU(),
        )
        self.sample_attention_block = nn.Sequential(
            nn.Linear(self.num_pathway + drug_for_pathway_size, self.num_pathway),
            nn.BatchNorm1d(self.num_pathway),
            nn.Tanh(),
            nn.Softmax(dim=1),
        )
        # Defined by the original but never called in forward. Kept so that
        # state_dict keys line up with published checkpoints.
        self.sample_multiplied_block = nn.Sequential(
            nn.BatchNorm1d(self.num_pathway), nn.ReLU(),
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

    def forward(self, input_feature, return_attention=False):
        gene_expression_list = input_feature[0]
        drug_fp = input_feature[1]
        dose = input_feature[2]

        drug_embed = self.drug_block(drug_fp)
        new_drug_embed = self.new_drug_block(drug_fp)

        attention_dots = []
        for idx, pathway in enumerate(self.pathway_list):
            gene_expression = gene_expression_list[idx]

            drug_gene_set_embed = self.drug_gene_set_blocks[pathway](new_drug_embed)
            gene_concat = torch.cat((gene_expression, drug_gene_set_embed), dim=1)
            gene_attention = self.gene_attention_blocks[pathway](gene_concat)
            attention_dot = batch_dot(gene_expression, gene_attention)
            attention_dots.append(self.gene_dot_blocks[pathway](attention_dot))

        drug_dense_embed = self.drug_dense_sample_block(new_drug_embed)
        drug_effected = attention_dots.copy()
        drug_effected.append(drug_dense_embed)

        gene_set_concat = torch.cat(attention_dots, dim=1)
        drug_effected_concat = torch.cat(drug_effected, dim=1)

        sample_attention = self.sample_attention_block(drug_effected_concat)
        sample_multiplied = torch.mul(gene_set_concat, sample_attention)

        total_concat = torch.cat([sample_multiplied, drug_embed], dim=1)
        concat_embed = self.concatenated_block(total_concat)
        final_embed = self.final_block(concat_embed)

        final_y_max = self.final_y_max(final_embed)
        final_y_min = self.final_y_min(final_embed)
        final_slope = self.final_slope(final_embed)
        final_ic50 = self.final_IC50(final_embed)

        final_1 = torch.sub(dose, final_ic50)
        final_2 = torch.mul(final_slope, final_1)
        final_neg = torch.mul(final_2, -1)
        final_sigmoid = torch.sigmoid(final_neg)
        final_scale = torch.sub(final_y_max, final_y_min)
        final_3 = torch.mul(final_scale, final_sigmoid)
        viability = torch.add(final_3, final_y_min)

        if return_attention:
            return viability, sample_attention
        return viability


class ReferenceCombinationTherapyModel(nn.Module):
    """Verbatim port of ``CombinationTherapyModel`` from
    00_CombinationtherapyUtils.ipynb.

    Structure, layer order, variable names and forward signature are the
    authors'. ``efficacy_relu`` and ``viability_relu`` are constructed and never
    called, exactly as published -- that is why Supplementary Data 3 contains
    negative viabilities. Do not tidy any of this: it is the baseline that
    defines correct.
    """

    def __init__(self):
        super().__init__()

        self.num_pathway = 186

        embedding_modules = []
        embedding_modules.append(nn.Linear(self.num_pathway, self.num_pathway))
        embedding_modules.append(nn.BatchNorm1d(self.num_pathway))
        embedding_modules.append(nn.ReLU())
        embedding_modules.append(nn.Tanh())
        embedding_modules.append(nn.Softmax(dim=1))
        self.embedding_network = nn.Sequential(*embedding_modules)

        monotherapy_modules = []
        monotherapy_modules.append(nn.Linear(self.num_pathway, 64))
        monotherapy_modules.append(nn.BatchNorm1d(64))
        monotherapy_modules.append(nn.ReLU())
        monotherapy_modules.append(nn.Linear(64, 16))
        monotherapy_modules.append(nn.BatchNorm1d(16))
        monotherapy_modules.append(nn.ReLU())
        monotherapy_modules.append(nn.Linear(16, 4))
        monotherapy_modules.append(nn.BatchNorm1d(4))
        monotherapy_modules.append(nn.ReLU())
        monotherapy_modules.append(nn.Linear(4, 1))
        self.monotherapy_network = nn.Sequential(*monotherapy_modules)

        self.monotherapy_coefficient_network = nn.Softmax(dim=1)

        synergy_modules = []
        synergy_modules.append(nn.Linear(2 * self.num_pathway, 64))
        synergy_modules.append(nn.BatchNorm1d(64))
        synergy_modules.append(nn.ReLU())
        synergy_modules.append(nn.Linear(64, 16))
        synergy_modules.append(nn.BatchNorm1d(16))
        synergy_modules.append(nn.ReLU())
        synergy_modules.append(nn.Linear(16, 4))
        synergy_modules.append(nn.BatchNorm1d(4))
        synergy_modules.append(nn.ReLU())
        synergy_modules.append(nn.Linear(4, 1))
        self.synergy_network = nn.Sequential(*synergy_modules)

        # Built by the original and never called in forward. Kept so state_dict
        # keys line up with published checkpoints.
        self.efficacy_relu = nn.ReLU()
        self.viability_relu = nn.ReLU()

    def forward(self, input_feature):
        pathway_attention1 = input_feature[0]
        pathway_attention2 = input_feature[1]
        viability1 = input_feature[2]
        viability2 = input_feature[3]

        efficacy1 = torch.sub(1, viability1)
        efficacy2 = torch.sub(1, viability2)

        embedding2to1 = self.embedding_network(pathway_attention2)
        embedding1to2 = self.embedding_network(pathway_attention1)

        pathway1with2 = torch.multiply(pathway_attention1, embedding2to1)
        pathway2with1 = torch.multiply(pathway_attention2, embedding1to2)

        processed_pathway1 = self.monotherapy_network(pathway1with2)
        processed_pathway2 = self.monotherapy_network(pathway2with1)

        processed_pathways = torch.cat([processed_pathway1, processed_pathway2], dim=1)
        efficacies = torch.cat([efficacy1, efficacy2], dim=1)
        processed_pathways_with_efficacies = torch.multiply(processed_pathways, efficacies)
        coefficient_mono = self.monotherapy_coefficient_network(processed_pathways_with_efficacies)
        monotherapy_effect = batch_dot(coefficient_mono, efficacies)

        pathway_attention1_effected = torch.mul(pathway_attention1, efficacy1)
        pathway_attention2_effected = torch.mul(pathway_attention2, efficacy2)

        pathway_concat = torch.cat([pathway_attention1_effected,
                                    pathway_attention2_effected], dim=1)
        synergy_effect = self.synergy_network(pathway_concat)

        combination_efficacy = torch.add(monotherapy_effect, synergy_effect)
        combination_viability = torch.sub(1, combination_efficacy)

        return coefficient_mono, synergy_effect, combination_viability
