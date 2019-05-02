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
N, D_X, D_Y = 20, 3, 1
sigma_obs = 0.1
X = 2.0 * torch.arange(N).float() / N - 1.0
X = torch.pow(X.unsqueeze(-1), torch.arange(D_X).float())
W = 0.3 * torch.randn(D_X)
Y = nonlin((X * W).sum(-1)) + sigma_obs * torch.randn(N)
Y = Y.unsqueeze(-1)
assert X.shape == (N, D_X)
assert Y.shape == (N, D_Y)

N_units = D_Y * D_X

def model():
    sigsq = pyro.sample("sigsq", dist.InverseGamma(3.0, 1.0))
    xi = pyro.sample("xi", dist.Normal(0.0, sigsq.sqrt()))
    with pyro.plate("units", N_units):
        w = pyro.sample("w", dist.Normal(0.0, sigsq.sqrt() / math.sqrt(D_X)))
        w_matrix = w.reshape(D_X, D_Y)

    with pyro.plate("data", N):
        z = nonlin(torch.matmul(X, w_matrix) + xi)
        pyro.sample("Y", dist.Normal(z, sigma_obs), obs=Y)
        return z

nuts_kernel = NUTS(model, adapt_step_size=True, jit_compile=True, adapt_mass_matrix=True,
                   full_mass=True, ignore_jit_warnings=True)

N_samples = 400
N_warmup = 300
hmc_posterior = MCMC(nuts_kernel, num_samples=N_samples, warmup_steps=N_warmup).run()
z_samples = hmc_posterior.marginal('_RETURN').empirical['_RETURN']._samples.squeeze(-1)

#w_samples = hmc_posterior.marginal('w').empirical['w']._samples
#sigsq_samples = hmc_posterior.marginal('sigsq').empirical['sigsq']._samples
#xi_samples = hmc_posterior.marginal('xi').empirical['xi']._samples

percentiles = np.percentile(z_samples.data.numpy(), [5.0, 95.0], axis=0)

plt.figure(figsize=(14,8))
plt.plot(X[:, 1].data.numpy(), Y[:, 0].data.numpy(), 'kx')
for k in range(0, N_samples, int(N_samples / 5)):
    plt.plot(X[:, 1].data.numpy(), z_samples[k, :].data.numpy(), color='k', linestyle='dashed', linewidth=0.5)
plt.fill_between(X[:, 1].data.numpy(), percentiles[0, :], percentiles[1, :], color='lightblue')
plt.savefig('out.pdf')
plt.close()
