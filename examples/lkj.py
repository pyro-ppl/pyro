from __future__ import absolute_import, division, print_function

import math

import torch

import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import math
from pyro.infer.mcmc import HMC, MCMC, NUTS
from pyro import poutine

"""
This simple example is intended to demonstrate how to use an LKJ prior with
a multivariate distribution.
"""

pyro.enable_validation(True)

# Fake, uncorrelated data
y = torch.randn(100, 3).to(dtype=torch.double)

def model(y):
    d = y.shape[1]
    N = y.shape[0]
    # Vector of variances for each of the d variables
    theta = pyro.sample("theta", dist.HalfCauchy(torch.full((d,), 1, dtype=torch.double)))
    # Lower cholesky factor of a correlation matrix
    L_omega = pyro.sample("L_omega", dist.CorrLCholeskyLKJPrior(d, torch.DoubleTensor([1])))
    # Lower cholesky factor of the covariance matrix
    L_Omega = torch.mm(torch.diag(theta.sqrt()), L_omega)

    with pyro.plate("observations", N) as n:
        obs = pyro.sample("obs", dist.MultivariateNormal(torch.zeros(d).to(dtype=torch.double), scale_tril=L_Omega), obs=y[n])
    return obs

nuts_kernel = NUTS(model, jit_compile=False)
posterior = MCMC(nuts_kernel, num_samples=200, warmup_steps=100, num_chains=1).run(y)
