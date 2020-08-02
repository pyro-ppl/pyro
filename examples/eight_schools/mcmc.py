# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import torch

import data
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(__debug__)
pyro.set_rng_seed(0)


def model(sigma):
    eta = pyro.sample('eta', dist.Normal(torch.zeros(data.J), torch.ones(data.J)))
    mu = pyro.sample('mu', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    tau = pyro.sample('tau', dist.HalfCauchy(scale=25 * torch.ones(1)))

    theta = mu + tau * eta

    return pyro.sample("obs", dist.Normal(theta, sigma))


def conditioned_model(model, sigma, y):
    return poutine.condition(model, data={"obs": y})(sigma)


def main(args):
    nuts_kernel = NUTS(conditioned_model, jit_compile=args.jit)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains)
    mcmc.run(model, data.sigma, data.y)
    mcmc.summary(prob=0.5)


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description='Eight Schools MCMC')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='number of MCMC samples (default: 1000)')
    parser.add_argument('--num-chains', type=int, default=1,
                        help='number of parallel MCMC chains (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='number of MCMC samples for warmup (default: 1000)')
    parser.add_argument('--jit', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
