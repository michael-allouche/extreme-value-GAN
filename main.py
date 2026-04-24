import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula
from scipy import stats as st
import torch

from models.utils import get_data_uqr
from models import dict_models


# ====================
#     CONFIGURATION
# ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Train and simulate extreme GAN models.")
    parser.add_argument("--model",      type=str,   default="gan",
                        choices=list(dict_models.keys()))
    parser.add_argument("--n_epochs",   type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=123)
    parser.add_argument("--verbose",    type=int,   default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return


# ====================
#       DATA
# ====================
def generate_data(n_data: int, dim_data: int, theta: float,
                  gamma: float, rho: float, seed: int = 123) -> np.ndarray:
    """
    Generate samples from a Gumbel copula with Burr margins.

    Parameters
    ----------
    n_data    : number of samples
    dim_data  : number of dimensions
    theta     : Gumbel copula dependence parameter (>= 1)
    gamma     : Burr tail index (> 0)
    rho       : Burr second-order parameter (< 0)
    seed      : random seed for reproducibility
    """
    marginals = [st.burr12(c=-rho / gamma, d=-1 / rho) for _ in range(dim_data)]
    copula = CopulaDistribution(
        copula=GumbelCopula(theta=theta, k_dim=dim_data),
        marginals=marginals
    )
    return copula.rvs(n_data, random_state=seed)


# ====================
#     VISUALISATION
# ====================
def plot_results(trainset, trainset_excess, X_sim_np, model_name, margin_i, margin_j):
    plt.figure(figsize=(9, 6))
    plt.scatter(trainset[:, margin_i], trainset[:, margin_j], label="Training data", alpha=0.3)
    plt.scatter(trainset_excess[:, margin_i],  trainset_excess[:, margin_j], label="Excess data", alpha=0.5)
    plt.scatter(X_sim_np[:, margin_i], X_sim_np[:, margin_j], label="Generated", alpha=0.5)
    plt.xlabel(f"Margin {margin_i + 1}", fontsize=15)
    plt.ylabel(f"Margin {margin_j + 1}", fontsize=15)
    plt.title(f"Simulated excess data — {model_name.upper()}", fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()



# ====================
#        MAIN
# ====================
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # --- Model Hyperparameters ---
    LATENT_DIM    = 10  # Latent Dimension of the Generator
    HIDDEN_DIM_G  = [30]  # Number of neurons per layer in the Generator
    HIDDEN_DIM_D  = [10, 10]  # Number of neurons per layer in the Discriminator
    LR_D          = 1e-4  # Learning rate Discriminator
    LR_G          = 1e-4  # Learning rate Generator
    NORMALIZATION = False  # If Max Normalization is applied on the data


    # --- Data Hyperparameters ---
    N_DATA        = 100000  # Data Sample size
    DIM_DATA      = 2  # Data Dimension
    ANCHOR_LEVELS = [0.05] * DIM_DATA  # Anchor levels delta_n for each margins
    # Burr parameters
    GAMMA = 0.5
    RHO = -1
    # Gumbel copula parameter
    THETA = 2

    # ====================
    #       DATA
    # ====================
    trainset = generate_data(N_DATA, DIM_DATA, THETA, GAMMA, RHO, seed=args.seed)
    trainset_excess, anchor_points = get_data_uqr(trainset, ANCHOR_LEVELS)

    # ===================
    #     MODELISATION
    # ===================
    model = dict_models[args.model](
        DIM_DATA, LATENT_DIM, HIDDEN_DIM_G, HIDDEN_DIM_D, lrD=LR_D, lrG=LR_G
    )
    model.train(trainset, args.n_epochs, args.batch_size, args.verbose,
                anchor_levels=ANCHOR_LEVELS, normalization=NORMALIZATION)

    # ===================
    #     SIMULATION
    # ===================
    with torch.no_grad():
        X_sim = model.simulate_excess(
            n_data=len(trainset_excess),
            anchor_levels=ANCHOR_LEVELS,
            anchor_points=anchor_points
        )
    X_sim_np = X_sim.numpy()

    # ====================
    #     VISUALISATION
    # ====================
    plot_results(trainset, trainset_excess, X_sim_np, args.model, margin_i=0, margin_j=1)
