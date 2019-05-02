from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
import math
import numpy as np

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def nonlin(x):
    return torch.tanh(x)

# Build a toy dataset.
N, D_X, D_Y, D_H = 85, 2, 1, 20
sigma_obs = 0.1
X = 2.0 * torch.arange(N).float() / N - 1.0
X = torch.pow(X.unsqueeze(-1), torch.arange(D_X).float() + 1.0)
W = 0.9 * torch.randn(D_X)
Y = torch.sin((X * W).sum(-1)) + sigma_obs * torch.randn(N)
Y = Y.unsqueeze(-1)
assert X.shape == (N, D_X)
assert Y.shape == (N, D_Y)


def model():
    with pyro.plate("activations1", D_H):
        sigsq1 = pyro.sample("sigsq1", dist.InverseGamma(3.0, 1.0))  # D_H
        xi1 = pyro.sample("xi1", dist.Normal(0.0, sigsq1.sqrt()))  # D_H
        with pyro.plate("inputs1", D_X):
            w1 = pyro.sample("w1", dist.Normal(0.0, sigsq1.sqrt() / math.sqrt(D_X)))  # D_X D_H
            z1 = nonlin(torch.matmul(X, w1) + xi1)  # N D_H

    with pyro.plate("activations2", D_Y):
        sigsq2 = pyro.sample("sigsq2", dist.InverseGamma(3.0, 1.0))
        xi2 = pyro.sample("xi2", dist.Normal(0.0, sigsq2.sqrt()))
        with pyro.plate("inputs2", D_H):
            w2 = pyro.sample("w2", dist.Normal(0.0, sigsq2.sqrt() / math.sqrt(D_H)))  # D_H D_Y
            z2 = nonlin(torch.matmul(z1, w2) + xi2)  # N D_Y

    with pyro.plate("data", N):
        pyro.sample("Y", dist.Normal(z2, sigma_obs), obs=Y)
        return z2


nuts_kernel = NUTS(model, adapt_step_size=True, jit_compile=True, adapt_mass_matrix=True,
                   full_mass=False, ignore_jit_warnings=True, max_tree_depth=6)

N_samples = 500
N_warmup = 200
hmc_posterior = MCMC(nuts_kernel, num_samples=N_samples, warmup_steps=N_warmup).run()

for site in ['w1', 'w2', 'xi1', 'xi2', 'sigsq1', 'sigsq2']:
    print("[%s] ESS" % site, hmc_posterior.marginal(site).diagnostics()[site]['n_eff'].data.numpy())


z_samples = hmc_posterior.marginal('_RETURN').empirical['_RETURN']._samples.squeeze(-1)

percentiles = np.percentile(z_samples.data.numpy(), [5.0, 95.0], axis=0)

plt.figure(figsize=(14,8))
plt.plot(X[:, 0].data.numpy(), Y[:, 0].data.numpy(), 'kx')
for k in range(0, N_samples, int(N_samples / 8)):
    plt.plot(X[:, 0].data.numpy(), z_samples[k, :].data.numpy(), color='k', linestyle='dashed', linewidth=0.5)
plt.fill_between(X[:, 0].data.numpy(), percentiles[0, :], percentiles[1, :], color='lightblue')
plt.savefig('out.pdf')
plt.close()
