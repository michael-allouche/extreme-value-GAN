import torch
import torch.nn as nn
from models.gan import GAN
from models.utils import BlockReLU
from scipy.special import expi
from pathlib import Path
import torch.nn.functional as F





class EVGAN(GAN):
    """Extreme Value GAN (EV-GAN).

        Extends the base GAN with a generator that incorporates a Tail-Index
        Function (TIF) correction, enabling accurate simulation extreme events.

        Reference
        ---------
        Allouche, Girard & Gobet (2022). EV-GAN: Simulation of extreme events
        with ReLU neural networks. JMLR 23, 1-39.
        https://jmlr.org/papers/v23/21-0609.html

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
        super(EVGAN, self).__init__(dim_data, latent_dim, hidden_dims_G, hidden_dims_D, float(lrD), float(lrG), **kwargs)

        torch.manual_seed(123)

        self.generator = Generator(dim_data, latent_dim, hidden_dims_G, self.device).to(self.device)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=float(lrG))

        self.pt_pathdir = Path("ckpt", "evgan")
        self.pt_pathdir.mkdir(parents=True, exist_ok=True)  # check if the directory exists

        return

class TIFinv(nn.Module):
    """Inverse Tail-Index Function (TIF) activation.

    Computes H_u^{-1}(x) = ((1 - u^2) / 2)^{-x}, which maps the
    network output x back to the extreme value scale. This function
    is used as the final activation of the EV-GAN generator.

    See equation (3) in Allouche et al. (2022).
    """
    def __init__(self):
        super(TIFinv, self).__init__()
        return

    def forward(self, inputs, x):
        """Apply the inverse TIF transformation.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise u of shape (batch_size, dim_data), values in (0, 1).
        x : torch.Tensor
            Network output (tail index) of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Transformed output ((1 - u^2) / 2)^{-x} of shape
            (batch_size, dim_data).
        """
        return torch.pow((1. - inputs.pow(2)) / 2, - x)


class Generator(nn.Module):
    """EV-GAN generator with learnable Tail-Index Function (TIF) correction.

        Extends the base MLP generator with two additive correction terms
        (correction_tif_1 and correction_tif_2) that encode the asymptotic
        tail behavior described by the TIF. The final activation is the
        inverse TIF (TIFinv), which maps the corrected network output to
        the extreme value scale.

        The six learnable parameters (var1–var6) correspond to:
          var1 : gamma  — tail index
          var2 : c0     — coefficient in Phi
          var3 : c1     — coefficient in Phi
          var4 : c2     — coefficient in Phi
          var5 : c3     — coefficient in Phi
          var6 : f'_TIF(0) — derivative of TIF at 0

        See Section 3 of Allouche et al. (2022).

        Parameters
        ----------
        dim_data : int
            Dimensionality of the output space.
        latent_dim : int
            Dimensionality of the latent input (must be >= dim_data).
        hidden_dims : list of int
            Hidden layer sizes.
        device : torch.device
            Compute device (needed for scipy round-trips in `li`).
        """
    def __init__(self, dim_data, latent_dim, hidden_dims, device):
        super(Generator, self).__init__()
        self.dim_data = dim_data
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.device = device

        self.activation_TIF = TIFinv()

        # Fully connected
        list_layers = []
        current_hdim = latent_dim
        for hidden_dim in hidden_dims:
            list_layers.append(BlockReLU(current_hdim, hidden_dim))
            current_hdim = hidden_dim
        # output layer
        list_layers.append(nn.Linear(current_hdim, dim_data))
        self.layers = nn.Sequential(*list_layers)

        # init constants for each dimension (component) on which we apply TIF correction
        # we allow c0 to be different from c3 for more degree of freedom
        self.var1 = nn.Parameter(torch.randn(1, self.dim_data), requires_grad=True)  # gamma
        self.var2 = nn.Parameter(torch.randn(1, self.dim_data), requires_grad=True)  # c0
        self.var3 = nn.Parameter(torch.randn(1, self.dim_data), requires_grad=True)  # c1
        self.var4 = nn.Parameter(torch.randn(1, self.dim_data), requires_grad=True)  # c2
        self.var5 = nn.Parameter(torch.randn(1, self.dim_data), requires_grad=True)  # c3
        self.var6 = nn.Parameter(torch.randn(1, self.dim_data), requires_grad=True)  # diff f^TIF (0)
        return

    def spline01(self, inputs):
        """Smooth interpolating spline with boundary conditions h(0)=0, h(1)=1.

        Computes p_01(t) = -4t^5 + 5t^4, the Hermite basis polynomial
        used to blend the TIF correction smoothly from 0 to 1.

        Parameters
        ----------
        inputs : torch.Tensor
            Values in [0, 1] of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Spline values of shape (batch_size, dim_data).
        """
        return (-4 * inputs**5) + (5 * inputs**4)

    def spline00(self, inputs):
        return inputs**3 - (2 * inputs**2) + inputs

    def correction_tif_1(self, inputs):
        """First TIF correction term encoding the asymptotic tail index.

        Computes p_01(u) * (gamma + Phi(u)), where gamma (var1) is the
        tail index and Phi encodes the second-order correction. NaN values
        near u=0 (from log(0)) are replaced by 0 via continuity extension.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise u in (0, 1) of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Correction values of shape (batch_size, dim_data).
        """
        correction = self.spline01(inputs) * (self.var1 + self.Phi(inputs))
        correction = torch.nan_to_num(correction, nan=0.0)  # extend by continuity at 0
        return correction

    def correction_tif_2(self, inputs):
        """Second TIF correction term encoding the derivative at 0.

        Computes p_00(u) * f'_TIF(0) (var6), ensuring the generator
        output has the correct slope at the boundary of the latent space.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise u in (0, 1) of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Correction values of shape (batch_size, dim_data).
                """
        return self.var6 * self.spline00(inputs)

    def Phi(self, inputs):
        """Second-order correction function Phi(u) based on the log-integral.

        Computes a linear combination of log and logarithmic-integral
        terms that captures the second-order regular variation of the
        underlying extreme value distribution. See equation (8) in
        Allouche et al. (2022).

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise u in (0, 1) of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            Second-order correction of shape (batch_size, dim_data).
        """
        log_term = torch.log(1. - inputs)
        li_term = self.li(1. - inputs)
        term1 = self.var2 / log_term
        term2 = self.var3 * li_term
        term3 = self.var4 * (( (1. - inputs) / log_term) - li_term)
        term4 = self.var5 * (( ((1 - inputs) * (log_term + 1)) / (2 * log_term ** 2)) - (li_term / 2))
        return term1 + term2 + term3 + term4

    def li(self, inputs):
        """Compute the logarithmic integral li(x) = Ei(ln(x)).

        Uses scipy.special.expi on CPU and moves the result back to the
        model device. This CPU round-trip is necessary because scipy does
        not support GPU tensors.

        Parameters
        ----------
        inputs : torch.Tensor
            Positive values of shape (batch_size, dim_data).

        Returns
        -------
        torch.Tensor
            li(x) values of shape (batch_size, dim_data) on self.device.
        """
        x = expi(torch.log(inputs).cpu().numpy())
        return torch.tensor(x).to(self.device)

    def forward(self, inputs):
        """Generate extreme-value samples with TIF correction.

        Runs the base MLP, adds both TIF correction terms, applies ReLU,
        then maps through the inverse TIF activation.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent noise u of shape (batch_size, latent_dim), values in (0, 1).
            The first dim_data components are used for the TIF corrections.

        Returns
        -------
        torch.Tensor
            Generated samples of shape (batch_size, dim_data).
        """
        network = self.layers(inputs)
        # this configuration of inputs allows to have latent_dim >= dim_data
        tif_corr_1 = self.correction_tif_1(inputs[:, :self.dim_data])
        tif_corr_2 = self.correction_tif_2(inputs[:, :self.dim_data])
        network_tif = F.relu(network + tif_corr_1 + tif_corr_2)
        return self.activation_TIF(inputs[:, :self.dim_data], network_tif)
