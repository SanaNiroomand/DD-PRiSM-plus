"""The paper's training objective: density-weighted MSE plus a correlation term.

    L_total = alpha * L_dMSE + beta * (1 - PCC)
    L_dMSE  = mean( (1 - gamma * density(y)) * (y - yhat)^2 )

Defaults are the paper's: alpha=1, beta=0.5, gamma=0.75.

The density weighting exists because cell-viability labels pile up near 1.0 --
most drugs at most doses do very little. Down-weighting the crowded region stops
the model from collapsing onto "predict no effect". The correlation term is
there for the same reason from the other side: an error-only objective drifts
toward the training mean, which scores well on MSE and is useless.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

# Labels are rounded to 2 decimals and looked up in a table spanning 0.00-1.50,
# which is the range left after the >1.5 outlier filter in preprocessing.
GRID_STEP = 0.01
GRID_MAX = 1.50
GRID_SIZE = int(round(GRID_MAX / GRID_STEP)) + 1


def estimate_density(viability):
    """Gaussian-KDE density of the training labels, min-max scaled to [0, 1].

    Returns a tensor indexed by ``round(y * 100)``. The published code keeps a
    Python dict and looks up one label at a time, which cannot run on a GPU; a
    lookup tensor is the same numbers with an index_select instead.
    """
    values = np.asarray(viability, dtype=np.float64).ravel()
    kernel = stats.gaussian_kde(values)
    grid = np.round(np.arange(0, GRID_MAX + GRID_STEP / 2, GRID_STEP), 2)
    density = np.array([kernel(point)[0] for point in grid])
    spread = density.max() - density.min()
    scaled = (density - density.min()) / spread if spread else np.zeros_like(density)
    return torch.tensor(scaled, dtype=torch.float32)


def density_weights(y_true, density_lut, gamma=0.75):
    index = torch.clamp((y_true.detach() / GRID_STEP).round().long(),
                        0, density_lut.numel() - 1)
    return 1 - gamma * density_lut.to(y_true.device)[index]


def pearson(y_pred, y_true):
    predicted = y_pred - y_pred.mean()
    actual = y_true - y_true.mean()
    denominator = predicted.pow(2).sum().sqrt() * actual.pow(2).sum().sqrt()
    return (predicted * actual).sum() / denominator.clamp_min(1e-12)


def rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


class CustomLoss(nn.Module):
    """Density-weighted MSE combined with a correlation penalty.

    Args:
        density_lut: from :func:`estimate_density` over the *training* labels.
            Deriving it from the whole dataset would leak the validation and
            test label distributions into training.
    """

    def __init__(self, density_lut, alpha=1.0, beta=0.5, gamma=0.75):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.register_buffer("density_lut", density_lut)

    def forward(self, y_pred, y_true):
        weight = density_weights(y_true, self.density_lut, self.gamma)
        weighted = torch.mean(weight * (y_pred - y_true) ** 2)
        plain = torch.mean((y_pred - y_true) ** 2)
        correlation = pearson(y_pred, y_true)
        loss = self.alpha * weighted + self.beta * (1 - correlation)
        return loss, plain, correlation
