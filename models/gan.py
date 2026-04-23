import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import BlockReLU, get_data_uqr
import time
from pathlib import Path
import warnings


class Generator(nn.Module):
    """MLP generator network mapping latent noise to the data space.

    The architecture is a sequence of BlockReLU hidden layers followed
    by a linear output layer and a final ReLU activation, ensuring that
    all generated values are non-negative (suitable for heavy-tailed
    positive-valued distributions).

    Parameters
    ----------
    dim_data : int
        Dimensionality of the output (data) space.
    latent_dim : int
        Dimensionality of the latent noise input.
    hidden_dims : list of int
        Number of neurons in each hidden layer, e.g. [64, 64].
    """
    def __init__(self, dim_data, latent_dim, hidden_dims):
        super(Generator, self).__init__()
        self.dim_data = dim_data
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        list_layers = []
        current_hdim = latent_dim
        for hidden_dim in hidden_dims:
            list_layers.append(BlockReLU(current_hdim, hidden_dim))
            current_hdim = hidden_dim

        # output layer
        list_layers.append(nn.Linear(current_hdim, dim_data))
        self.layers = nn.Sequential(*list_layers)
        return

    def forward(self, inputs):
        """Map latent noise to the data space.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise tensor of shape (batch_size, latent_dim),
            sampled from U(0, 1)^{latent_dim}.

        Returns
        -------
        torch.Tensor
            Generated samples of shape (batch_size, dim_data),
            with all values >= 0 due to the final ReLU.
        """
        return F.relu(self.layers(inputs))

class Discriminator(nn.Module):
    """MLP discriminator network distinguishing real from generated samples.

    The architecture is a sequence of BlockReLU hidden layers followed
    by a linear output layer and a sigmoid activation, producing
    per-dimension probabilities of a sample being real.

    Parameters
    ----------
    dim_data : int
        Dimensionality of the input (data) space.
    hidden_dims : list of int
        Number of neurons in each hidden layer, e.g. [64, 64].
    """
    def __init__(self, dim_data, hidden_dims):
        super(Discriminator, self).__init__()
        self.dim_data = dim_data
        self.hidden_dims = hidden_dims

        list_layers = []
        current_hdim = dim_data
        for hidden_dim in hidden_dims:
            list_layers.append(BlockReLU(current_hdim, hidden_dim))
            current_hdim = hidden_dim

        # output layer
        list_layers.append(nn.Linear(current_hdim, dim_data))
        list_layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*list_layers)

    def forward(self, inputs):
        """Compute per-dimension real/fake probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Data samples of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Per-dimension probabilities in (0, 1) of shape
            (batch_size, dim_data).
        """
        return self.layers(inputs)

class GAN(nn.Module):
    def __init__(self, dim_data, latent_dim, hidden_dims_G, hidden_dims_D,
                 lrD=1e-3, lrG=1e-3,  **kwargs):
        """Vanilla GAN model.

        Implements a standard GAN training loop with binary cross-entropy
        losses.

        Can be used standalone or subclassed by EVGAN, FLExceedGAN, and
        LVExceedGAN for extreme-value-specific architectures.

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
            Learning rate for the discriminator optimizer. Default: 1e-3.
        lrG : float, optional
            Learning rate for the generator optimizer. Default: 1e-3.
        **kwargs
            Absorbed for compatibility with subclass constructors.
        """
        super(GAN, self).__init__()

        torch.manual_seed(123)
        self.device = self.get_device()

        self.dim_data = dim_data
        self.latent_dim = latent_dim
        self.hidden_dims_G = hidden_dims_G
        self.hidden_dims_D = hidden_dims_D
        self.lrG = lrG
        self.lrD = lrD

        # Anchor points for each margin
        # -----------------------------
        # Refers to the probability sequences \delta_n in order to simulate Y(\delta_n) = X | X > F_X^{-1}(1-\delta_n)
        self.anchor_levels = None
        self.anchor_points = None # Refers to the anchor points F_X^{-1}(1-\delta_n)
        # -----------------------------

        self.Xmax = None  # Per-margin maxima used for normalization.
        self.normalization = None  # Whether max normalization is active.

        self.generator = Generator(dim_data, latent_dim, hidden_dims_G).to(self.device)
        self.discriminator = Discriminator(dim_data, hidden_dims_D).to(self.device)

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=lrG)
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lrD)

        self.pt_pathdir = Path("ckpt", "gan")
        self.pt_pathdir.mkdir(parents=True, exist_ok=True)  # check if the directory exists

        return


    # ===============================================================================================
    #                                       Training
    # ===============================================================================================
    def train(self, data_train, n_epochs, batch_size, verbose, anchor_levels=None, k_train_D=1, normalization=False):
        """Train the GAN on data in the defined upper quadrant region.

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
        anchor_levels : array-like of float or None, optional
            Exceedance probability delta_n per margin. If None, the full
            dataset is used (no UQR filtering). Default: None.
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
        if anchor_levels is None:  # Consider all the data
            self.anchor_levels = torch.zeros(self.dim_data, device=self.device, dtype=torch.float32)
        else:
            self.anchor_levels = torch.tensor(anchor_levels, dtype=torch.float32).to(self.device)
            # TODO check is len(self.anchor_levels) == self.dim_data

        early_stopping = 0
        old_loss = 0

        if normalization:  # min max normalization
            data_train = self.data_normalization(data_train)
        # Get data in the upper-right quadrant region
        data_train, anchor_points= get_data_uqr(data_train, self.anchor_levels)

        # TODO: raise error message if no data in data_train
        self.anchor_points = torch.tensor(anchor_points, device=self.device, dtype=torch.float32)

        # Save Generator/Discriminator anchor data
        self.generator.anchor_points = self.anchor_points
        self.generator.anchor_levels = self.anchor_levels
        self.discriminator.anchor_levels = self.anchor_levels



        # For validation / evaluation
        # ----------------------------
        noise_valid = torch.rand(data_train.shape[0], self.latent_dim, device=self.device)  # fix the noise valid
        # ----------------------------

        while noise_valid.min() == 0:
            noise_valid = torch.rand(data_train.shape[0], self.latent_dim, device=self.device)  # uniform U(0,1)

        trainset = torch.utils.data.DataLoader(torch.tensor(data_train, dtype=torch.float32),
                                               batch_size=batch_size, shuffle=True,
                                               worker_init_fn=lambda _: np.random.seed(0))
        try:
            for epoch in range(1, n_epochs + 1):
                start_time = time.time()
                for X_train in trainset:
                    X_train = X_train.to(self.device)

                    # ===============================================================================
                    #                               Train Discriminator
                    # ===============================================================================
                    for _ in range(k_train_D):
                        noise = torch.rand(batch_size, self.latent_dim, device=self.device)  # uniform U[0,1)
                        while noise.min() == 0:
                            noise = torch.rand(batch_size, self.latent_dim, device=self.device)  # uniform U[0,1)

                        self.discriminator.zero_grad()
                        generated_data = self.generator(noise)
                        real_output = self.discriminator(X_train)
                        fake_output = self.discriminator(generated_data)
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

                    generated_data = self.generator(noise)
                    fake_output = self.discriminator(generated_data)
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


    def discriminator_loss(self, real_output, fake_output):
        """Compute the binary cross-entropy loss for the discriminator.

        The total loss is:
            L_D = -E[log D(x)] - E[log(1 - D(G(z)))]

        Parameters
        ----------
        real_output : torch.Tensor
            Discriminator output on real samples, shape (batch, dim_data).
        fake_output : torch.Tensor
            Discriminator output on generated samples, shape (batch, dim_data).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        real_loss = F.binary_cross_entropy(input=real_output, target=torch.ones_like(real_output, device=self.device))  #  - log(D(x))
        fake_loss = F.binary_cross_entropy(input=fake_output, target=torch.zeros_like(fake_output, device=self.device))  # - log(1 - D(G(z)))
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        """Compute the binary cross-entropy loss for the generator.

        The generator is trained to maximize log D(G(z)), equivalently:
            L_G = -E[log D(G(z))]

        Parameters
        ----------
        fake_output : torch.Tensor
            Discriminator output on generated samples, shape (batch, dim_data).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return F.binary_cross_entropy(input=fake_output, target=torch.ones_like(fake_output, device=self.device))  # - log(D(G(z)))

    def data_normalization(self, data):
        """Apply component-wise max normalization to the data.

        Divides each feature by its maximum value observed in the training
        set (stored in self.Xmax), mapping all values to (0, 1].

        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_samples, dim_data).

        Returns
        -------
        np.ndarray
            Normalized data of shape (n_samples, dim_data).
        """
        return data / self.Xmax.reshape(1,-1)

    def inv_data_normalization(self, x):
        """Invert the component-wise max normalization.

        Multiplies each feature by the stored per-margin maximum (self.Xmax),
        mapping values from (0, 1] back to the original scale.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor of shape (n_samples, dim_data).

        Returns
        -------
        torch.Tensor
            Rescaled tensor of shape (n_samples, dim_data).
        """
        return x * self.Xmax

    # ===============================================================================================
    #                                       Simulations
    # ===============================================================================================
    @torch.no_grad()
    def simulate(self, n_data, noise=None, seed=0):
        """Generate samples from the trained generator.

        Parameters
        ----------
        n_data : int
            Number of samples to generate.
        noise : torch.Tensor or None, optional
            Pre-sampled latent noise of shape (n_data, latent_dim).
            If None, fresh uniform noise is sampled. Default: None.
        seed : int, optional
            Random seed used when noise is None. Default: 0.

        Returns
        -------
        torch.Tensor
            Generated samples of shape (n_data, dim_data) on CPU.
            Rescaled to original units if normalization was active.
        """
        if noise is None:
            torch.manual_seed(seed)
            noise = torch.rand((n_data, self.latent_dim), device=self.device)
            while noise.min() == 0:
                noise = torch.rand((n_data, self.latent_dim), device=self.device)
        if self.normalization:
            return self.generator(noise).cpu() * self.Xmax
        else:
            return self.generator(noise).cpu()

    @torch.no_grad()
    def simulate_excess(self, n_data, noise=None, seed=0, **kwargs):
        """Simulate exactly n_data points in the upper quadrant region.

        Uses acceptance-rejection: generates batches from the Generator and
        retains only samples that exceed the stored anchor points on every
        margin. Repeats until n_data accepted samples are collected.

        Parameters
        ----------
        n_data : int
            Number of excess samples to return.
        noise : torch.Tensor or None, optional
            Pre-sampled latent noise of shape (n_data, latent_dim).
            If None, fresh uniform noise is sampled. Default: None.
        seed : int, optional
            Random seed used when noise is None. Default: 0.

        Returns
        -------
        torch.Tensor
            Accepted excess samples of shape (n_data, dim_data).
            Rescaled to original units if normalization was active.
        """
        if kwargs:
            warnings.warn(f"simulate_excess is using the anchor levels and the anchor points defined in the model")

        n_simulated_points = n_data
        if noise is None:  # if no input noise, simulate some
            torch.manual_seed(seed)
            noise = torch.rand((n_data, self.latent_dim), device=self.device)
            while noise.min() == 0:
                noise = torch.rand((n_data, self.latent_dim), device=self.device)

        X = self.generator(noise).cpu()
        X_excess = X[np.all([X[:,d] > self.anchor_points[d] for d in range(len(self.anchor_points))], axis=0)]
        while X_excess.shape[0] < n_data:
            n_rejected = (n_data - X_excess.shape[0])
            n_simulated_points += n_rejected
            noise = torch.rand((n_rejected, self.latent_dim), device=self.device)
            while noise.min() == 0:
                noise = torch.rand((n_rejected, self.latent_dim), device=self.device)
            X = self.generator(noise).cpu()
            X_excess = torch.cat([X_excess, X[np.all([X[:, d] > self.anchor_points[d] for d in range(len(self.anchor_points))], axis=0)]])

        X_out = X_excess[:n_data]
        return X_out * self.Xmax if self.normalization else X_out

# ==========================================================================================================

    @staticmethod
    def get_device() -> torch.device:
        """Select the best available compute device.

        Returns
        -------
        torch.device
            The selected device: cuda:0, mps, or cpu.
        """
        if torch.cuda.is_available():
            return torch.device("cuda:0")  # on GPU
        else:
            return torch.device("cpu")  # on CPU


