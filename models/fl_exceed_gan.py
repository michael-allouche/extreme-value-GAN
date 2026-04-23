import torch
import torch.nn as nn
from models.gan import GAN
from models.utils import BlockELU, BlockReLU
from pathlib import Path
import torch.nn.functional as F



class Generator(nn.Module):
    """Fixed-Level ExceedGAN generator.

        Maps latent uniform noise to excess samples above a fixed threshold,
        conditioning on the anchor level delta_n and the corresponding
        anchor point F_X^{-1}(1 - delta_n).

        The input is transformed to exponential random variables via -log(u),
        and the output is rescaled by the anchor points and exponentiated,
        ensuring generated samples lie above the threshold by construction.

        See Section 4.1 of Allouche et al. (2026), Extremes 1-23.

        It takes advantage of the eLU parametrization, which is known to be the natural basis functions to
        approximate extreme quantiles.
        See Allouche, Girard & Gobet (2024).
        Estimation of extreme quantiles from heavy-tailed distributions with neural networks. Statistics and Computing 34(12).
        https://link.springer.com/article/10.1007/s11222-023-10331-2

        Parameters
        ----------
        dim_data : int
            Dimensionality of the data.
        latent_dim : int
            Dimensionality of the latent noise input.
        hidden_dims : list of int
            Hidden layer sizes.
        anchor_levels : torch.Tensor or None
            Exceedance probabilities delta_n, shape (dim_data,).
            Set to None at init; updated after training via get_data_uqr.
        anchor_points : torch.Tensor or None
            Empirical quantiles F_X^{-1}(1 - delta_n), shape (dim_data,).
            Set to None at init; updated after training via get_data_uqr.
        device : torch.device
            Compute device.
        """
    def __init__(self, dim_data, latent_dim, hidden_dims, anchor_levels, anchor_points, device):
        super(Generator, self).__init__()
        self.dim_data = dim_data
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.anchor_levels = anchor_levels
        self.anchor_points = anchor_points
        self.device = device

        # Fully connected
        list_layers = []
        current_hdim = latent_dim + self.dim_data
        for hidden_dim in hidden_dims:
            list_layers.append(BlockELU(current_hdim, hidden_dim))
            current_hdim = hidden_dim

        # output layer
        list_layers.append(nn.Linear(current_hdim, self.dim_data))
        self.layers = nn.Sequential(*list_layers)
        return

    def forward(self, inputs):
        """Generate samples conditioned on the fixed anchor level.

        Concatenates the latent noise with -log(delta_n) (exponential
        transform of the anchor level), passes through the MLP, and
        rescales the output.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise of shape (batch_size, latent_dim), values in (0, 1).

        Returns
        -------
        torch.Tensor
            Generated excess samples of shape (batch_size, dim_data),
            guaranteed to be positive.
        """
        u = torch.ones(inputs.shape[0], self.dim_data, device=self.device) * self.anchor_levels
        network = self.layers(-torch.log(torch.cat([inputs, u], dim=1)))  # transform to exponential random variables
        return self.anchor_points * torch.exp(F.relu(network))


class Discriminator(nn.Module):
    """Fixed-Level ExceedGAN discriminator.

    Distinguishes real excess samples from generated ones.

    Parameters
    ----------
    dim_data : int
        Dimensionality of the data.
    hidden_dims : list of int
        Hidden layer sizes.
    anchor_levels : torch.Tensor or None
        Exceedance probabilities delta_n, shape (dim_data,).
    device : torch.device
        Compute device.
    """
    def __init__(self, dim_data, hidden_dims, device):
        super(Discriminator, self).__init__()
        self.dim_data = dim_data
        self.hidden_dims = hidden_dims
        self.device = device

        list_layers = []
        current_hdim = dim_data
        for hidden_dim in hidden_dims:
            list_layers.append(BlockReLU(current_hdim, hidden_dim))
            current_hdim = hidden_dim

        # output layer
        list_layers.append(nn.Linear(current_hdim, self.dim_data))
        list_layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*list_layers)

    def forward(self, inputs):
        return self.layers(inputs)

class FLExceedGAN(GAN):  # inherits from GAN
    """Fixed-Level Exceed GAN (FL-ExceedGAN).

    Trains a GAN to simulate multivariate extreme events above a fixed
    threshold level delta_n. The generator is
    conditioned on -log(delta_n) to encode the exceedance level. Anchor
    points are determined from data during training.

    Inherits training loop and simulation utilities from GAN, overriding
    only the generator and discriminator architectures.

    Reference
    ---------
    Allouche, Girard & Gobet (2026).
    ExceedGAN: simulation above extreme thresholds using Generative Adversarial Networks
    Extremes 1-23. https://link.springer.com/article/10.1007/s10687-026-00528-9

    Parameters
    ----------
    dim_data : int
        Dimensionality of the data.
    latent_dim : int
        Dimensionality of the generator's latent noise input.
    hidden_dims_G : list of int
        Hidden layer sizes for the generator.
    hidden_dims_D : list of int
        Hidden layer sizes for the discriminator.
    lrD : float, optional
        Learning rate for the discriminator. Default: 1e-3.
    lrG : float, optional
        Learning rate for the generator. Default: 1e-3.
    """
    def __init__(self, dim_data, latent_dim, hidden_dims_G, hidden_dims_D, lrD=1e-3, lrG=1e-3, **kwargs):
        super(FLExceedGAN, self).__init__(dim_data, latent_dim, hidden_dims_G, hidden_dims_D, float(lrD), float(lrG), **kwargs)

        torch.manual_seed(123)
        # Anchor points for each margin
        # -----------------------------
        # Refers to the probability sequences \delta_n in order to simulate Y(\delta_n) = X | X > F_X^{-1}(1-\delta_n)
        self.anchor_levels = None
        # Refers to the anchor points F_X^{-1}(1-\delta_n)
        self.anchor_points = None
        # -----------------------------

        self.generator = Generator(dim_data, latent_dim, hidden_dims_G, self.anchor_levels, self.anchor_points, self.device).to(self.device)
        self.discriminator = Discriminator(dim_data, hidden_dims_D, self.device).to(self.device)

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=float(lrG))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=float(lrD))

        self.pt_pathdir = Path("ckpt", "fl_exceed_gan")
        self.pt_pathdir.mkdir(parents=True, exist_ok=True)  # check if the directory exists

        return