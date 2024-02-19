import torch
from functorch.dim import dims

import pyro
import pyro.distributions as dist

i, j = dims(2)
mu = torch.ones(3, 2)
normal = dist.Normal(mu[i, j], 1, validate_args=True)
import pdb

pdb.set_trace()

with pyro.plate("z_plate", 5) as z:
    import pdb

    pdb.set_trace()
    print(f"z = {z}")

f, c, d = dims(3)
p = torch.ones(5, 4, 3)
i = torch.ones(3).long()[d]  # c
pi = p[f, i]
pc = p[f, c]
pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import constraints
from tqdm import tqdm

import pyro
from pyro.distributions import *
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam

assert pyro.__version__.startswith("1.8.6")
pyro.set_rng_seed(0)

device = torch.device("cuda")

data = torch.cat(
    (
        MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([50]),
        MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([50]),
        MultivariateNormal(torch.tensor([1.5, 2]), torch.eye(2)).sample([50]),
        MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)).sample([50]),
    )
)

data = data.to(device)
# plt.scatter(data[:, 0], data[:, 1])
# plt.title("Data Samples from Mixture of 4 Gaussians")
# plt.show()
N = data.shape[0]
num_particles = 10


########################################
def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


########################################


def model(data):
    with pyro.plate("beta_plate", T - 1, device=device):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T, device=device):
        mu = pyro.sample(
            "mu",
            MultivariateNormal(
                torch.zeros(2, device=device), 5 * torch.eye(2, device=device)
            ),
        )

    with pyro.plate("data", N, device=device) as idx:
        # dim=-4 is an enumeration dim for z (e.g. T = 6 clusters)
        # dim=-3 is a particle vectorization (e.g. num_particles = 10 particles)
        # dim=-2 is allocated for "data" plate (1 value broadcasted over a batch)
        # dim=-1 is allocated as event dimension (2 values)
        z = pyro.sample("z", Categorical(mix_weights(beta).unsqueeze(-2))[idx])
        pyro.sample(
            "obs", MultivariateNormal(mu[z], torch.eye(2, device=device)), obs=data[idx]
        )


########################################
def guide(data):
    kappa = pyro.param(
        "kappa",
        lambda: Uniform(
            torch.tensor(0.0, device=device), torch.tensor(2.0, device=device)
        ).sample([T - 1]),
        constraint=constraints.positive,
    )
    tau = pyro.param(
        "tau",
        lambda: MultivariateNormal(
            torch.zeros(2, device=device), 3 * torch.eye(2, device=device)
        ).sample([T]),
    )
    phi = pyro.param(
        "phi",
        lambda: Dirichlet(1 / T * torch.ones(T, device=device)).sample([N]),
        constraint=constraints.simplex,
    )

    with pyro.plate("beta_plate", T - 1, device=device):
        q_beta = pyro.sample("beta", Beta(torch.ones(T - 1, device=device), kappa))

    with pyro.plate("mu_plate", T, device=device):
        q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2, device=device)))

    with pyro.plate("data", N, device=device):
        z = pyro.sample("z", Categorical(phi))
