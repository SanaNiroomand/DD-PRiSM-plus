"""Combination therapy model: split a drug pair's effect into three parts.

    E(C, D1, D2, d1, d2) = alpha * E(C, D1, d1) + beta * E(C, D2, d2) + gamma
    V = 1 - E

alpha and beta weight each drug's own effect; gamma is what the pair does beyond
them, which is the synergy the paper is after. Inputs are the frozen Monotherapy
model's outputs: each drug's predicted viability and its 186-dim pathway
attention.

Verified against Supplementary Data 3, which publishes all five quantities for
2,556 combinations: the identity above reproduces the authors' own viabilities
to 1.5e-07, and alpha + beta = 1 to 1e-07.
"""

import torch
import torch.nn as nn


def _mlp(in_features):
    """The [., 64, 16, 4, 1] stack used for both coefficient networks."""
    return nn.Sequential(
        nn.Linear(in_features, 64), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Linear(64, 16), nn.BatchNorm1d(16), nn.ReLU(),
        nn.Linear(16, 4), nn.BatchNorm1d(4), nn.ReLU(),
        nn.Linear(4, 1),
    )


class CombinationTherapyModel(nn.Module):
    """Args:
        num_pathways: 186 for KEGG legacy.
        bound_output: clamp efficacy and viability to be non-negative.

            The published model builds ``efficacy_relu`` and ``viability_relu``
            in ``__init__`` and never calls them, so its output is unbounded.
            Supplementary Data 3 shows the consequence: 573 of 2,556 published
            rows have negative viability, down to -0.78, which cannot happen --
            viability is a surviving cell fraction.

            Default False reproduces the paper. Set True to fix it, at the cost
            of no longer matching their numbers exactly.
    """

    def __init__(self, num_pathways=186, bound_output=False):
        super().__init__()
        self.num_pathways = num_pathways
        self.bound_output = bound_output

        # How drug 2's pathway profile reweights drug 1's, and vice versa.
        # Note ReLU before Tanh: that is the published order, and it means the
        # Tanh output is [0, 1) rather than (-1, 1).
        self.embedding_network = nn.Sequential(
            nn.Linear(num_pathways, num_pathways), nn.BatchNorm1d(num_pathways),
            nn.ReLU(), nn.Tanh(), nn.Softmax(dim=1),
        )
        # Shared between the two drugs -- one network applied twice, so the
        # model cannot learn an order-dependent preference here.
        self.monotherapy_network = _mlp(num_pathways)
        self.synergy_network = _mlp(2 * num_pathways)

    def forward(self, attention1, attention2, viability1, viability2):
        efficacy1 = 1 - viability1
        efficacy2 = 1 - viability2

        weighted1 = attention1 * self.embedding_network(attention2)
        weighted2 = attention2 * self.embedding_network(attention1)

        scores = torch.cat([self.monotherapy_network(weighted1),
                            self.monotherapy_network(weighted2)], dim=1)
        efficacies = torch.cat([efficacy1, efficacy2], dim=1)

        # Scaling the scores by the efficacies before the softmax makes a drug
        # that does nothing unable to claim weight in the combination.
        coefficients = torch.softmax(scores * efficacies, dim=1)
        monotherapy_effect = (coefficients * efficacies).sum(dim=1, keepdim=True)

        synergy = self.synergy_network(
            torch.cat([attention1 * efficacy1, attention2 * efficacy2], dim=1))

        efficacy = monotherapy_effect + synergy
        if self.bound_output:
            efficacy = torch.relu(efficacy)
        viability = 1 - efficacy
        if self.bound_output:
            viability = torch.relu(viability)

        return coefficients, synergy, viability
