# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from functools import partial
import torch
from torch.distributions import constraints
import numpy as np

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_eig
import pyro.contrib.gp as gp
from pyro.contrib.oed.glmm import (
    zero_mean_unit_obs_sd_lm, group_assignment_matrix, analytic_posterior_cov
)

from gp_bayes_opt import GPBayesOptimizer

"""
Example builds on the Bayesian regression tutorial [1]. It demonstrates how
to estimate the average posterior entropy (APE) under a model and use it to
make an optimal decision about experiment design.

The context is a Gaussian linear model in which the design matrix `X` is a
one-hot-encoded matrix with 2 columns. This corresponds to the simplest form
of an A/B test. Assume no data has yet be collected. The aim is to find the optimal
allocation of participants to the two groups to maximise the expected gain in
information from actually performing the experiment.

For details of the implementation of average posterior entropy estimation, see
the docs for :func:`pyro.contrib.oed.eig.vi_eig`.

We recommend the technical report from Long Ouyang et al [2] as an introduction
to optimal experiment design within probabilistic programs.

To optimize the APE (which is required to be minimized) we used Gaussian Process
based Bayesian Optimization. See the BO tutorial [3] for details of optimizing noisy
and expensive-to-compute functions in pyro.

[1] ["Bayesian Regression"](http://pyro.ai/examples/bayesian_regression.html)
[2] Long Ouyang, Michael Henry Tessler, Daniel Ly, Noah Goodman (2016),
    "Practical optimal experiment design with probabilistic programs",
    (https://arxiv.org/abs/1608.05046)
[3] ["Bayesian Optimization"](http://pyro.ai/examples/bo.html)
"""

# Set up regression model dimensions
N = 100  # number of participants
p = 2    # number of features
prior_sds = torch.tensor([10., 2.5])

# Model and guide using known obs_sd
model, guide = zero_mean_unit_obs_sd_lm(prior_sds)


def estimated_ape(ns, num_vi_steps):
    designs = [group_assignment_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
    X = torch.stack(designs)
    est_ape = vi_eig(
        model,
        X,
        observation_labels="y",
        target_labels="w",
        vi_parameters={
            "guide": guide,
            "optim": optim.Adam({"lr": 0.05}),
            "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
            "num_steps": num_vi_steps},
        is_parameters={"num_samples": 1},
        eig=False
    )
    return est_ape


def true_ape(ns):
    """Analytic APE"""
    true_ape = []
    prior_cov = torch.diag(prior_sds**2)
    designs = [group_assignment_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
    for i in range(len(ns)):
        x = designs[i]
        posterior_cov = analytic_posterior_cov(prior_cov, x, torch.tensor(1.))
        true_ape.append(0.5*torch.logdet(2*np.pi*np.e*posterior_cov))
    return torch.tensor(true_ape)


def main(num_vi_steps, num_bo_steps, seed):

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    est_ape = partial(estimated_ape, num_vi_steps=num_vi_steps)
    est_ape.__doc__ = "Estimated APE by VI"

    estimators = [true_ape, est_ape]
    noises = [0.0001, 0.25]
    num_acqs = [2, 10]

    for f, noise, num_acquisitions in zip(estimators, noises, num_acqs):
        X = torch.tensor([25., 75.])
        y = f(X)
        gpmodel = gp.models.GPRegression(
            X, y, gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(10.)),
            noise=torch.tensor(noise), jitter=1e-6)
        gpbo = GPBayesOptimizer(constraints.interval(0, 100), gpmodel,
                                num_acquisitions=num_acquisitions)
        pyro.clear_param_store()
        for i in range(num_bo_steps):
            result = gpbo.get_step(f, None, verbose=True)

        print(f.__doc__)
        print(result)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="A/B test experiment design using VI")
    parser.add_argument("-n", "--num-vi-steps", nargs="?", default=5000, type=int)
    parser.add_argument('--num-bo-steps', nargs="?", default=5, type=int)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    main(args.num_vi_steps, args.num_bo_steps, args.seed)
