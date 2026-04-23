import torch
import torch.nn as nn
from models.gan import GAN
from models.utils import BlockReLU, BlockELU, get_data_uqr
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import time


class Generator(nn.Module):
    """Level-Varying ExceedGAN generator.

    See Section 4.2 of Allouche et al. (2026), Extremes 1-23.

    Takes advantage of the eLU parametrization, which is known to be the natural basis functions to
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
    device : torch.device
        Compute device.
        """
    def __init__(self, dim_data, latent_dim, hidden_dims, device):
        super(Generator, self).__init__()
        self.dim_data = dim_data
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.device = device

        # Fully connected
        list_layers = []
        current_hdim = latent_dim + self.dim_data
        for hidden_dim in hidden_dims:
            list_layers.append(BlockELU(current_hdim, hidden_dim))
            current_hdim = hidden_dim

        # output layer
        list_layers.append(nn.Linear(current_hdim, dim_data))
        self.layers = nn.Sequential(*list_layers)
        return


    def forward(self, inputs, anchor_levels, anchor_points):
        """Generate excess samples conditioned on a batch of anchor levels.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise of shape (batch_size, latent_dim), values in (0, 1).
        anchor_levels : torch.Tensor
            Per-sample exceedance probabilities of shape (batch_size, dim_data).
        anchor_points : torch.Tensor
            Per-sample empirical quantiles of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Generated excess samples of shape (batch_size, dim_data),
            guaranteed to be positive.
        """
        network = self.layers(-torch.log(torch.cat([inputs, anchor_levels], dim=1)))  # transform to exponential random variables
        return anchor_points * torch.exp(F.relu(network))

class Discriminator(nn.Module):
    """Level-Varying ExceedGAN discriminator.

    Distinguishes real excess samples from generated ones, conditioning
    on the anchor level by appending -log(delta_n) to each sample.
    This makes the discriminator aware of which threshold was used,
    allowing it to operate across varying levels.

    Parameters
    ----------
    dim_data : int
        Dimensionality of the data.
    hidden_dims : list of int
        Hidden layer sizes.
    """
    def __init__(self, dim_data, hidden_dims):
        super(Discriminator, self).__init__()
        self.dim_data = dim_data
        self.hidden_dims = hidden_dims

        list_layers = []
        current_hdim = dim_data + dim_data
        for hidden_dim in hidden_dims:
            list_layers.append(BlockReLU(current_hdim, hidden_dim))
            current_hdim = hidden_dim

        # output layer
        list_layers.append(nn.Linear(current_hdim, dim_data))
        list_layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*list_layers)

    def forward(self, inputs, anchor_levels):
        """Compute per-dimension real/fake probabilities.

        Appends -log(delta_n) to the input before discrimination,
        making the network threshold-aware.

        Parameters
        ----------
        inputs : torch.Tensor
            Data samples of shape (batch_size, dim_data).
        anchor_levels : torch.Tensor
            Per-sample exceedance probabilities of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Per-dimension probabilities in (0, 1) of shape (batch_size, dim_data).
        """
        return self.layers(torch.cat([inputs, -torch.log(anchor_levels)], dim=1))


class LVExceedGAN(GAN):
    """Level-Varying Exceed GAN (LV-ExceedGAN).

    Trains a GAN to simulate multivariate extreme events at varying
    threshold levels. At each training step, anchor levels are sampled
    uniformly from (0, 1 - support_a)^D, making the generator and
    discriminator aware of arbitrary exceedance probabilities.

    This allows simulation at any threshold level seen during training,
    without retraining.

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
    support_a : float, optional
        Upper bound parameter for the anchor level support. Anchor levels
        are sampled uniformly from (0, 1 - support_a). Must be in (0, 1).
        Default: 0.5.
    """
    def __init__(self, dim_data, latent_dim, hidden_dims_G, hidden_dims_D, lrD=1e-3, lrG=1e-3, support_a=0.5, **kwargs):
        super(LVExceedGAN, self).__init__(dim_data, latent_dim, hidden_dims_G, hidden_dims_D, float(lrD), float(lrG), **kwargs)

        torch.manual_seed(123)
        # Anchor points for each margin
        # -----------------------------
        # Refers to the probability sequences \delta_n in order to simulate Y(\delta_n) = X | X > F_X^{-1}(1-\delta_n)
        self.anchor_levels = None
        # Refers to the anchor points F_X^{-1}(1-\delta_n)
        self.anchor_points = None
        self.support_a = support_a
        # -----------------------------

        self.generator = Generator(dim_data, latent_dim, hidden_dims_G, self.device).to(self.device)
        self.discriminator = Discriminator(dim_data, hidden_dims_D).to(self.device)

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=float(lrG))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=float(lrD))

        self.pt_pathdir = Path("ckpt", "lv_exceed_gan")
        self.pt_pathdir.mkdir(parents=True, exist_ok=True)  # check if the directory exists

        return

    def train(self, data_train, n_epochs, batch_size, verbose, k_train_D=1, normalization=False, **kwargs):
        """Train the LV-ExceedGAN with randomly sampled anchor levels.

        At each epoch, a full mini-batch is constructed by independently
        drawing a random anchor level for each sample, then sampling one
        training point from the corresponding upper quadrant region. This
        makes the model learn to generate excess distributions for any
        threshold level in (0, 1 - support_a).

        Parameters
        ----------
        data_train : np.ndarray
            Training dataset of shape (n_samples, dim_data).
        n_epochs : int
            Maximum number of training epochs.
        batch_size : int
            Number of samples per mini-batch.
        verbose : int
            Print training statistics every `verbose` epochs.
            Early stopping is also checked every `verbose` epochs.
        k_train_D : int, optional
            Number of discriminator updates per generator update.
            Default: 1.
        normalization : bool, optional
            If True, apply component-wise max normalization to data_train
            before training. Default: False.
        """
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.Xmax = torch.tensor(data_train.max(axis=0), dtype=torch.float32, device=self.device)
        self.normalization = normalization

        support_conditional_law = (0, 1 - self.support_a) # Hypercube (0, 1-a)^D, with a\in(0,1)

        early_stopping = 0
        old_loss = 0

        if normalization:  # min max normalization
            data_train = self.data_normalization(data_train)

        # trainset = torch.tensor(data_train)
        try:
            for epoch in range(1, n_epochs + 1):
                start_time = time.time()
                X_train = []
                anchor_levels_batchs = []
                anchor_points_batchs = []
                for _ in range(batch_size): # Build a batch of points in the upper quadrant region
                    # Simulate an anchor level in [r1, r2]: U(0,1)*(r1- r2) + r2
                    anchor_level_batch = (torch.rand(size=(1, self.dim_data), device=self.device) * (support_conditional_law[0] - support_conditional_law[1]) + support_conditional_law[1]).ravel()
                    # Simulate a point in the excess region of the training set
                    X_uqr, anchor_point_batch = get_data_uqr(data_train, anchor_level_batch)

                    while X_uqr.shape[0] == 0:  # keep sampling if no training point in the upper quadrant region
                        anchor_level_batch = (torch.rand(size=(1, self.dim_data), device=self.device) * (support_conditional_law[0] - support_conditional_law[1]) + support_conditional_law[1]).ravel()
                        X_uqr, anchor_point_batch = get_data_uqr(data_train, anchor_level_batch)

                    X_uqr = torch.tensor(X_uqr, dtype=torch.float32, device=self.device)

                    anchor_levels_batchs.append(anchor_level_batch)
                    anchor_points_batchs.append(torch.tensor(anchor_point_batch, dtype=torch.float32, device=self.device))
                    random_point = torch.randint(0, X_uqr.shape[0], (1,1))[0][0]  # draw an integer from uniform dist
                    X_train.append(X_uqr[random_point,:]) # select a random point from the excess ones
                X_train = torch.stack(X_train).to(self.device)
                anchor_levels_batchs = torch.stack(anchor_levels_batchs).to(self.device)
                anchor_points_batchs = torch.stack(anchor_points_batchs).to(self.device)

                # ===============================================================================
                #                               Train Discriminator
                # ===============================================================================
                for _ in range(k_train_D):
                    noise = torch.rand(batch_size, self.latent_dim, device=self.device)  # uniform U[0,1)
                    while noise.min() == 0:
                        noise = torch.rand(batch_size, self.latent_dim, device=self.device)  # uniform U[0,1)

                    self.discriminator.zero_grad()
                    generated_data = self.generator(noise, anchor_levels_batchs, anchor_points_batchs)
                    real_output = self.discriminator(X_train, anchor_levels_batchs)
                    fake_output = self.discriminator(generated_data, anchor_levels_batchs)
                    lossD = self.discriminator_loss(real_output, fake_output)
                    lossD.backward()  # compute gradient dL/dD
                    self.optimizerD.step()  # update weights
                # ===============================================================================

                # ===============================================================================
                #                               Train Generator
                # ===============================================================================
                noise = torch.rand(batch_size, self.latent_dim, device=self.device)
                while noise.min() == 0:
                    noise = torch.rand(batch_size, self.latent_dim, device=self.device)  # uniform U[0,1)

                generated_data = self.generator(noise, anchor_levels_batchs, anchor_points_batchs)
                fake_output = self.discriminator(generated_data, anchor_levels_batchs)
                self.generator.zero_grad()
                lossG = self.generator_loss(fake_output)
                lossG.backward() # compute gradient dL/dG
                self.optimizerG.step() # update weights
                # ===============================================================================

                if epoch % verbose == 0:  # action every verbose epochs
                    # =================================================================
                    #     ADD CRITERIA EVALUATION HERE USING noise_valid
                    #           BEST MODELS CAN BE SAVED IN self.pt_pathdir
                    # =================================================================
                    time_epoch = (time.time() - start_time)
                    print("Epoch {} ({:.2f} sec): Loss Generator={:.4f}, Loss Discriminator={:.4f}".format(
                        epoch, time_epoch, lossG.detach().cpu().numpy(), lossD.detach().cpu().numpy()))

                    new_loss = np.round(lossG.detach().cpu().numpy(), 8) + np.round(lossD.detach().cpu().numpy(),8)
                    if old_loss == new_loss:
                        early_stopping += 1
                        if early_stopping == verbose:  # stop if the loss stays flat
                            print("Early Stopping")
                            break
                    else:
                        early_stopping = 0
                    old_loss = new_loss

        except RuntimeError as e:
            print("Runtime Error at epoch {}: {}".format(epoch, e))
            return
        print("------ END TRAINING ------")
        return


    # ===============================================================================================
    #                                       Simulations
    # ===============================================================================================
    @torch.no_grad()
    def simulate_excess(self, n_data, anchor_levels, anchor_points, noise=None, seed=0, **kwargs):
        """Simulate excess samples at a specified threshold level.

        Unlike the base GAN's acceptance-rejection approach, LV-ExceedGAN
        generates samples that exceed the threshold by construction, using
        the generator conditioned on the provided anchor level and points.

        Parameters
        ----------
        n_data : int
            Number of samples to generate.
        anchor_levels : array-like of float
            Exceedance probabilities delta_n per margin, shape (dim_data,).
            Must lie within the support seen during training: (0, 1 - support_a).
        anchor_points : np.ndarray
            Empirical quantiles F_X^{-1}(1 - delta_n) per margin,
            shape (dim_data,). Typically obtained from get_data_uqr.
        noise : torch.Tensor or None, optional
            Pre-sampled latent noise of shape (n_data, latent_dim).
            If None, fresh uniform noise is sampled. Default: None.
        seed : int, optional
            Random seed used when noise is None. Default: 0.
        **kwargs
            Absorbed for API compatibility; not used.

        Returns
        -------
        torch.Tensor
            Generated excess samples of shape (n_data, dim_data) on CPU.
            Rescaled to original units if normalization was active.
        """
        """Simulate in the upper quadrant region by construction"""
        if self.normalization:
            anchor_points = self.data_normalization(anchor_points)
        anchor_levels_batch = torch.tensor(anchor_levels, device=self.device).expand(n_data, -1) #torch.tensor(anchor_levels) * torch.ones(n_data, len(anchor_levels))
        anchor_points_batch = torch.tensor(anchor_points, device=self.device).expand(n_data, -1) #torch.tensor(anchor_points) * torch.ones(n_data, len(anchor_points))
        if noise is None:
            torch.manual_seed(seed)
            noise = torch.rand((n_data, self.latent_dim), device=self.device)
            while noise.min() == 0:
                noise = torch.rand((n_data, self.latent_dim), device=self.device)
        X_sim = self.generator(noise, anchor_levels_batch, anchor_points_batch).cpu()
        return X_sim * self.Xmax if self.normalization else X_sim

