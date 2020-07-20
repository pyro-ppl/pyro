# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS

"""
This simple example is intended to demonstrate how to use an LKJ prior with
a multivariate distribution.

It generates entirely random, uncorrelated data, and then attempts to fit a correlation matrix
and vector of variances.
"""


def model(y):
    d = y.shape[1]
    N = y.shape[0]
    options = dict(dtype=y.dtype, device=y.device)
    # Vector of variances for each of the d variables
    theta = pyro.sample("theta", dist.HalfCauchy(torch.ones(d, **options)))
    # Lower cholesky factor of a correlation matrix
    eta = torch.ones(1, **options)  # Implies a uniform distribution over correlation matrices
    L_omega = pyro.sample("L_omega", dist.LKJCorrCholesky(d, eta))
    # Lower cholesky factor of the covariance matrix
    L_Omega = torch.mm(torch.diag(theta.sqrt()), L_omega)
    # For inference with SVI, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)

    # Vector of expectations
    mu = torch.zeros(d, **options)

    with pyro.plate("observations", N):
        obs = pyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y)
    return obs


def main(args):
    y = torch.randn(args.n, args.num_variables).to(dtype=torch.double)
    if args.cuda:
        y = y.cuda()
    nuts_kernel = NUTS(model, jit_compile=False, step_size=1e-5)
    MCMC(nuts_kernel, num_samples=args.num_samples,
         warmup_steps=args.warmup_steps, num_chains=args.num_chains).run(y)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="Demonstrate the use of an LKJ Prior")
    parser.add_argument("--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--n", nargs="?", default=500, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument("--num-variables", nargs='?', default=5, type=int)
    parser.add_argument("--warmup-steps", nargs='?', default=100, type=int)
    parser.add_argument("--rng_seed", nargs='?', default=0, type=int)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    pyro.set_rng_seed(args.rng_seed)
    # Enable validation checks
    pyro.enable_validation(__debug__)

    # work around with the error "RuntimeError: received 0 items of ancdata"
    # see https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823
    torch.multiprocessing.set_sharing_strategy("file_system")

    main(args)
