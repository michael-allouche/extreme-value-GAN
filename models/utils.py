import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockReLU(nn.Module):
    """Block of fully-connected layer followed by a ReLU activation function.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    """
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(inputs))


class BlockELU(nn.Module):
    """Block of fully-connected layer followed by a eLU activation function.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    """
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        return F.elu(self.fc(inputs))


def get_data_uqr(X, anchor_levels):
    """
    Filter a dataset to its upper quadrant region (UQR).

    Returns observations where every marginal exceeds its corresponding
    empirical quantile at level (1 - delta_n), as defined by anchor_levels.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataset of shape (n_samples, n_features).
    anchor_levels : array-like of float
        Exceedance probability sequence delta_n in (0, 1) for each margin.
        A value of 0 keeps all the data for that margin.

    Returns
    -------
    X_uqr : pd.DataFrame
        Subset of X in the upper quadrant region.
    anchor_points : np.ndarray
        Empirical quantiles F_X^{-1}(1 - delta_n) for each margin.
    """
    n_data = X.shape[0]
    anchor_points = np.array([
        np.sort(X[:, j])[np.maximum(0, int(np.around((1 - anchor_levels[j]) * n_data))-1)]
        for j in range(len(anchor_levels))
    ])
    X_uqr = X[np.all([X[:, j] > anchor_points[j] for j in range(len(anchor_levels))], axis=0)]
    return X_uqr, anchor_points