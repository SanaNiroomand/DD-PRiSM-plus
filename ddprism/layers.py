"""Grouped layers: many small per-pathway modules evaluated as one batched op.

Each layer here stands in for P independent ``nn.Linear`` / ``nn.BatchNorm1d``
modules. Given the same weights, outputs match the per-module version on the
masked (real) slots to floating-point tolerance.
"""

import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    """P independent Linear layers, evaluated with a single ``bmm``.

    Input is (B, P, in_features), output is (B, P, out_features). Groups are
    padded to a common width; ``in_counts``/``out_counts`` give each group's
    real width so initialisation can use the correct fan-in.
    """

    def __init__(self, num_groups, in_features, out_features,
                 in_counts=None, out_counts=None):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.in_counts = in_counts
        self.out_counts = out_counts

        self.weight = nn.Parameter(torch.empty(num_groups, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(num_groups, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.Linear's default init reduces to U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        # for both weight and bias. Padded columns stay at zero so they cannot
        # contribute even if a caller forgets to mask the input.
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()
            for group in range(self.num_groups):
                fan_in = (self.in_counts[group] if self.in_counts
                          else self.in_features)
                n_out = (self.out_counts[group] if self.out_counts
                         else self.out_features)
                bound = 1.0 / (fan_in ** 0.5)
                self.weight[group, :n_out, :fan_in].uniform_(-bound, bound)
                self.bias[group, :n_out].uniform_(-bound, bound)

    def forward(self, x):
        # (B, P, I) -> (P, B, I) @ (P, I, O) -> (P, B, O) -> (B, P, O)
        out = torch.bmm(x.transpose(0, 1), self.weight.transpose(1, 2))
        return out.transpose(0, 1) + self.bias

    def forward_shared(self, x):
        """Same layers, but every group sees the same (B, in_features) input.

        Avoids materialising a (B, P, in_features) copy of identical rows.
        """
        return torch.einsum("bi,goi->bgo", x, self.weight) + self.bias


class GroupedBatchNorm1d(nn.Module):
    """BatchNorm over (B, P, F), treating every (pathway, feature) as a channel.

    Equivalent to P separate ``nn.BatchNorm1d(n_p)`` modules. Statistics are
    taken over the batch dimension only, so each real channel sees exactly the
    same numbers it would in the per-module version -- padding cannot leak in.
    """

    def __init__(self, num_groups, num_features, mask=None,
                 eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_groups, num_features))
        self.bias = nn.Parameter(torch.zeros(num_groups, num_features))
        self.register_buffer("running_mean", torch.zeros(num_groups, num_features))
        self.register_buffer("running_var", torch.ones(num_groups, num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "mask", mask.to(torch.bool) if mask is not None else None)

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.num_batches_tracked += 1
                batch = x.shape[0]
                # Normalisation uses the biased variance, the running estimate
                # uses the unbiased one. This asymmetry is PyTorch's, and it
                # has to be copied or eval-mode outputs drift apart.
                unbiased = var * batch / (batch - 1) if batch > 1 else var
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased)
        else:
            mean = self.running_mean
            var = self.running_var

        out = (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
        if self.mask is not None:
            out = out * self.mask
        return out
