# Extreme-value-GANs

This repository is dedicated to the implementation of two published papers, in joint work
with [Emmanuel Gobet](https://emmanuel-gobet.github.io/index.html) and [Stéphane Girard](https://mistis.inrialpes.fr/people/girard/)
:

- **EV-GAN: Simulation of extreme events
with ReLU neural networks** — [Allouche, Girard & Gobet, JMLR 2022](https://www.jmlr.org/papers/volume23/21-0663/21-0663.pdf)
- **ExceedGAN: Simulation above extreme thresholds using
Generative Adversarial Networks** — [Allouche, Girard & Gobet, Extremes 2026](https://inria.hal.science/hal-05044516v3/document)

Four generative models are implemented:

| Model                   | Key `--model` | Paper |
|-------------------------|---------------|-------|
| Standard GAN (baseline) | `gan` | Goodfellow et al. (2014) |
| EV-GAN                  | `evgan` | EV-GAN, JMLR 2022 |
| Fixed-Level ExceedGAN   | `fl_exceed_gan` | ExceedGAN, Extremes 2026 |
| Level-Varying ExceedGAN | `lv_exceed_gan` | ExceedGAN, Extremes 2026 |

---

## Introduction

Let $$ X = (X^{(1)}, \ldots, X^{(D)}) $$ be a $$ D $$-dimensional random vector with
heavy-tailed marginals. The **exceedance distribution** above a componentwise threshold
$$ u_n = F_X^{-1}(1 - \delta_n) $$ is the conditional distribution

$$Y(\delta_n) \;=\; X \;\Big|\; X > u_n$$

where the inequality is componentwise. The **upper quadrant region** at level $$ \delta_n $$
is defined as

$$\mathcal{Q}(\delta_n) = \left\{x \in \mathbb{R}^D : x^{(m)} > F_{X^{(m)}}^{-1}(1-\delta_n^{(m)}), \; m = 1,\ldots,D\right\}.$$

The figure below illustrates this on a bivariate dataset (log-log scale). The full dataset
is shown in blue. The two dashed rectangles delimit the upper quadrant regions at two
different threshold levels: the **green** rectangle corresponds to $$ \delta_n = (0.1, 0.1)^\top $$
(moderate extreme region) and the **red** rectangle to $$ \delta_n = (0.05, 0.05)^\top $$
(deeper extreme region). The goal of the models in this repository is to accurately
simulate new observations within these extreme regions.

![data.jpg](imgs/data.jpg)

A classical GAN with bounded latent input cannot reproduce heavy-tailed margins. The three
models below address this by adapting the generator parametrization to the extreme-value
framework.

---

## Models

### 1. GAN (Baseline)
![GAN.png](imgs/GAN.png)
A standard GAN with:

- **Generator**: MLP with ReLU activations mapping $$ z \sim U([0,1]^{d'}) $$ to $$ \mathbb{R}^D $$
- **Discriminator**: MLP with Sigmoid output
- **Loss**: Binary cross-entropy (BCE)
  - $$ \mathcal{L}_D = -\mathbb{E}[\log D(X)] - \mathbb{E}[\log(1-D(G(z)))] $$
  - $$ \mathcal{L}_G = -\mathbb{E}[\log D(G(z))] $$
- **Sampling**: Acceptance-rejection to obtain samples in $$ \mathcal{Q}(\delta_n) $$

> ⚠️ The classical GAN is included as a baseline. It is **not adapted** to heavy-tailed
> distributions and performs poorly in extreme regions.

---

### 2. EV-GAN

> Allouche, Girard & Gobet. *EV-GAN: Simulation of extreme events with ReLU neural networks.*
> **JMLR 23** (2022) 1–39.

#### Key idea

The **Tail-Index Function (TIF)** transforms the heavy-tailed quantile function
$$ q_X(u) \to +\infty $$ as $$ u \to 1 $$ into a bounded, continuous function on $$ [0,1] $$:

$$f^{\mathrm{TIF}}(u) = \frac{-\log\, q_X(1-(1-u)\eta)}{\log(1-u^2) - \log 2}, \qquad u \in [0,1),$$

with $$ f^{\mathrm{TIF}}(u) \to \gamma $$ as $$ u \to 1 $$. A ReLU network can then
approximate $$ f^{\mathrm{TIF}} $$ via the Universal Approximation Theorem. To further
reduce bias, a **Corrected TIF** (CTIF) subtracts 6 universal correction functions
$$ e_1, \ldots, e_6 $$ that encode the second-order behavior:

$$f^{\mathrm{CTIF}}(u) = f^{\mathrm{TIF}}(u) - \sum_{k=1}^{6} \kappa_k \, e_k(u)$$

where $$ e_1(u) = g(u) = -4u^5 + 5u^4 $$ and $$ e_2(u) = h(u) = u^3 - 2u^2 + u $$
are Hermite splines, and $$ e_{k+3}(u) = g(u)\,\Phi_k(u) $$ for $$ k = 0,\ldots,3 $$
involve the **logarithmic integral** $$ \mathrm{li}(x) = \int_0^x \frac{dt}{\log t} $$
via $$ \Phi_k $$ functions.

The final **EV-GAN generator** for dimension $$ m $$ is:

$$G_\psi^{\mathrm{TIF},(m)}(z) = H_{z^{(m)}}^{-1}\!\left(\sum_{j=1}^{J} a_j^{(m)}\,\sigma\!\left(\sum_{i=1}^{d'} w_j^{(i)} z^{(i)} + b_j\right) + \sum_{k=1}^{6} \kappa_k^{(m)}\,e_k(z^{(m)})\right)$$

where $$ H_u^{-1}(x) = \left(\dfrac{1-u^2}{2}\right)^{-x} $$ is the **inverse TIF activation**,
and $$ \sigma = \mathrm{ReLU} $$. The latent dimension satisfies $$ d' \geq D $$.

#### Architecture (Figure 2, EV-GAN — $$ d' = 3 $$, $$ D = 2 $$)
![EVGAN.png](imgs/EVGAN.png)
---

### 3. ExceedGAN

> Allouche, Girard & Gobet. *ExceedGAN: Simulation above extreme thresholds using GANs.*
> **Extremes** (2026). DOI: 10.1007/s10687-026-00528-9. HAL: hal-05044516.


#### Key idea

Simulates directly in $$ \mathcal{Q}(\delta_n) $$ for a threshold $$ \delta_n $$.
The generator approximates the log-spacing function using **eLU activation functions**
(instead of ReLU) for higher-order bias correction. The latent input and anchor level are
log-transformed before entering the network. The output is rescaled by the empirical anchor
point $$ X^{(m)}_{n-k+1,n} $$.

#### Architecture (Figure 2, ExceedGAN — $$ q = 3 $$, $$ D = 2 $$)

![ExcessGAN.png](imgs/ExcessGAN.png)

#### 3.1 Fixed-Level ExceedGAN (FL-ExceedGAN)

#### Key idea

Based on Lemma 3.1: $$ Y(\delta_n) \overset{d}{=} F_X^{-1}(1 - \delta_n Z) $$ with
$$ Z \sim U([0,1]) $$. The log-spacing function

$$f(x_1, x_2) = \log U_X(e^{x_1+x_2}) - \log U_X(e^{x_2}) = \gamma x_1 + \varphi(x_1, x_2)$$

is approximated by a NN with $$ J(J-1) $$ **eLU** neurons:

$$\varphi^{\mathrm{NN}}_J(x_1, x_2;\theta) = \sum_{i=1}^{J(J-1)/2} w_i^{(1)} \left\{\sigma_E\!\left(w_i^{(2)} x_1 + w_i^{(3)} x_2\right) - \sigma_E\!\left(w_i^{(4)} x_2\right)\right\}$$

where $$ \sigma_E(x) = x \cdot \mathbf{1}(x>0) + (e^x-1)\cdot\mathbf{1}(x\leq 0) $$ is the
**eLU activation**, which captures the higher-order bias of the tail. The generator for
margin $$ m $$ is:

$$G^{\mathrm{EX},(m)}(z) = X^{(m)}_{n-k^{(m)}+1,n} \cdot \exp\!\left(\sigma_R\!\left[f^{\mathrm{NN}}_J\!\left(\log(1/z), \log(1/\delta_n)\right)\right]\right)$$

where $$ X^{(m)}_{n-k+1,n} $$ is the empirical anchor point (order statistic) and
$$ k^{(m)} = \lfloor n\delta_n^{(m)} \rfloor $$.


Inputs are **log-transformed**: $$ -\log(z^{(j)}) = \log(1/z^{(j)}) $$ and
$$ -\log(\delta_n^{(m')}) = \log(1/\delta_n^{(m')}) $$. This maps the $$ (0,1) $$ latent
space to $$ (0, +\infty) $$, suitable for approximating the log-spacing function.


#### 3.2. Level-Varying ExceedGAN (LV-ExceedGAN)

Extends FL-ExceedGAN to **any threshold level** with a single trained model. The anchor
level $$ \delta_n $$ is treated as a conditioning variable, sampled uniformly from
$$ (0, 1-a)^D $$ at each training step (with $$ a \in (0,1) $$ chosen by the user).

The training uses a conditional GAN objective:

$$\arg\min_\theta \max_\varphi \left(\mathbb{E}_{p_U}\!\left[\mathbb{E}_{p_{Y(u_n)}}\!\left[\log D^\mathrm{EX}_\varphi(Y, \delta_n)\right]\right] + \mathbb{E}_{p_U}\!\left[\mathbb{E}_{p_Z}\!\left[\log\left(1 - D^\mathrm{EX}_\varphi\!\left(G^\mathrm{EX}_\theta(Z,\delta_n),\delta_n\right)\right)\right]\right]\right)$$

where $$ p_U $$ is the uniform distribution on $$ (0, 1-a)^D $$.

At simulation time, **any** threshold $$ \delta_n \in (0, 1-a)^D $$ can be provided —
no retraining is needed.

---

## Numerical Results

The plots below compare all four models on synthetic data simulated from a **bivariate
Gumbel copula** (dependence parameter $$ \mu = 2 $$, Kendall's $$ \tau = 0.5 $$) with
**Burr Type XII margins** (second-order parameters $$ (\rho_1, \rho_2) = (-1, -3) $$),
across three tail indices $$ \gamma \in \{0.3, 0.5, 0.9\} $$.

In each panel, **black crosses** represent the ground-truth test exceedances and **colored
dots** are generated samples from the corresponding model. Both axes are on a log scale.

### Extreme region $$ \delta_n = (0.05, 0.05)^\top $$

![simulations_005.png](imgs/simulations_005.png)

### Moderate region $$ \delta_n = (0.1, 0.1)^\top $$

![simulations_01.png](imgs/simulations_01.png)

**Key observations:**

- **GAN** (blue): systematically underestimates the tail — generated samples are too
  concentrated and fail to cover the upper extreme region, especially as $$ \gamma $$
  increases.
- **EV-GAN** (orange): significantly better tail coverage than the baseline GAN, but
  shows some dispersion mismatch for heavier tails ($$ \gamma = 0.9 $$).
- **FL-ExceedGAN** (green): best overall alignment with the test data across all values
  of $$ \gamma $$, correctly reproducing both the marginal spread and the dependence
  structure.
- **LV-ExceedGAN** (red): competitive with FL-ExceedGAN and particularly well-calibrated
  in the more extreme region $$ \delta_n = (0.05, 0.05)^\top $$, where it achieves the
  best coverage of the upper tail.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/exceedgan.git
cd exceedgan

# Install dependencies
pip install torch torchvision torchaudio   # PyTorch (MPS included on macOS)
pip install numpy scipy statsmodels matplotlib
```

**Minimum requirements:**

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 1.12 (MPS support) |
| NumPy | ≥ 1.22 |
| SciPy | ≥ 1.8 |

---

## Usage

### Quick start

```bash
# Train a Fixed-Level ExceedGAN on synthetic Gumbel-Burr data
python main.py --model fl_exceed_gan --n_epochs 10000 --verbose 100

# Train a Level-Varying ExceedGAN
python main.py --model lv_exceed_gan --n_epochs 10000 --batch_size 32

# Run the baseline GAN
python main.py --model gan --n_epochs 1000
```

### Available arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gan` | One of `gan`, `evgan`, `fl_exceed_gan`, `lv_exceed_gan` |
| `--n_epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--seed` | `123` | Global random seed |
| `--verbose` | `100` | Print loss every N epochs |

### Python API

```python
from models import dict_models
from models.utils import get_data_uqr
import torch, numpy as np

# Generate / load data
data = ...  # np.ndarray of shape (n_samples, D)
ANCHOR_LEVELS = [0.05, 0.05]

# Filter to upper quadrant region
data_excess, anchor_points = get_data_uqr(data, ANCHOR_LEVELS)

# Build and train the model
model = dict_models["fl_exceed_gan"](
    dim_data=2, latent_dim=10,
    hidden_dims_G=[30], hidden_dims_D=[10, 10],
    lrD=1e-4, lrG=1e-4
)
model.train(data, n_epochs=10000, batch_size=32, verbose=100,
            anchor_levels=ANCHOR_LEVELS, normalization=False)

# Simulate exceedances
with torch.no_grad():
    X_sim = model.simulate_excess(
        n_data=len(data_excess),
        anchor_levels=ANCHOR_LEVELS,
        anchor_points=anchor_points
    )
```


## Repository Structure

```
.
├── imgs/
│   ├── GAN.png                  # GAN architecture diagram
│   ├── EVGAN.png                # EV-GAN architecture diagram
│   └── ExcessGAN.png            # ExceedGAN architecture diagram
├── models/
│   ├── __init__.py              # Model registry (dict_models)
│   ├── utils.py                 # Shared blocks (BlockReLU, BlockELU) and get_data_uqr
│   ├── gan.py                   # Base GAN
│   ├── evgan.py                 # EV-GAN
│   ├── fl_exceed_gan.py         # Fixed-Level ExceedGAN
│   └── lv_exceed_gan.py         # Level-Varying ExceedGAN
├── ckpt/                        # Saved checkpoints
├── main.py                      # Training and simulation example
└── README.md
```

---

## References

```bibtex
@article{allouche2022evgan,
  title   = {{EV-GAN}: Simulation of extreme events with {R}e{LU} neural networks},
  author  = {Allouche, Michaël and Girard, Stéphane and Gobet, Emmanuel},
  journal = {Journal of Machine Learning Research},
  volume  = {23},
  pages   = {1--39},
  year    = {2022},
  url     = {https://jmlr.org/papers/v23/21-0663.html}
}

@article{allouche2026exceedgan,
  title   = {{ExceedGAN}: Simulation above extreme thresholds using
             Generative Adversarial Networks},
  author  = {Allouche, Michaël and Girard, Stéphane and Gobet, Emmanuel},
  journal = {Extremes},
  year    = {2026},
  doi     = {10.1007/s10687-026-00528-9},
  note    = {HAL: hal-05044516}
}
```

---

## License

Distributed under the **CC BY-NC-ND 4.0** license.
See [LICENSE](LICENSE) for details.