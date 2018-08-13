import argparse
from functools import partial
import torch
from torch.distributions import constraints

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_ape
import pyro.contrib.gp as gp

from gp_bayes_opt import GPBayesOptimizer
from models.bayes_linear import (
    bayesian_linear_model, normal_inv_gamma_guide, group_assignment_matrix,
    analytic_posterior_entropy
)

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
the docs for :func:`pyro.contrib.oed.eig.vi_ape`.

We recommend the technical report from Long Ouyang et al [2] as an introduction
to optimal experiment design within probabilistic programs.

[1] ["Bayesian Regression"](http://pyro.ai/examples/bayesian_regression.html)
[2] Long Ouyang, Michael Henry Tessler, Daniel Ly, Noah Goodman (2016),
    "Practical optimal experiment design with probabilistic programs",
    (https://arxiv.org/abs/1608.05046)
"""

# Set up regression model dimensions
N = 100  # number of participants
p = 2    # number of features
prior_sds = torch.tensor([10., 2.5])

# Model and guide using known obs_sd
model = partial(bayesian_linear_model, w_mean=torch.tensor(0.),
                w_sqrtlambda=1/prior_sds, obs_sd=torch.tensor(1.))
guide = partial(normal_inv_gamma_guide, obs_sd=torch.tensor(1.))


def estimated_ape(ns, num_vi_steps):
    designs = [group_assignment_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
    X = torch.stack(designs)
    est_ape = vi_ape(
        model,
        X,
        observation_labels="y",
        vi_parameters={
            "guide": guide,
            "optim": optim.Adam({"lr": 0.05}),
            "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
            "num_steps": num_vi_steps},
        is_parameters={"num_samples": 1}
    )
    return est_ape


def true_ape(ns):
    """Analytic APE"""
    true_ape = []
    prior_cov = torch.diag(prior_sds**2)
    designs = [group_assignment_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
    for i in range(len(ns)):
        x = designs[i]
        true_ape.append(analytic_posterior_entropy(prior_cov, x, torch.tensor(1.)))
    return torch.tensor(true_ape)


def main(num_vi_steps, num_acquisitions, num_bo_steps):

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    est_ape = partial(estimated_ape, num_vi_steps=num_vi_steps)
    est_ape.__doc__ = "Estimated APE by VI"

    estimators = [true_ape, est_ape]
    noises = [0.001, 0.1]

    for f, noise in zip(estimators, noises):
        X = torch.tensor([25., 75.])
        y = f(X)
        gpmodel = gp.models.GPRegression(
            X, y, gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(10.)),
            noise=torch.tensor(noise), jitter=1e-6)
        gpmodel.optimize(loss=TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss)
        gpbo = GPBayesOptimizer(constraints.interval(0, 100), gpmodel,
                                num_acquisitions=num_acquisitions)
        for i in range(num_bo_steps):
            pyro.clear_param_store()
            result = gpbo.get_step(f, None)

        print(f.__doc__)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test experiment design using VI")
    parser.add_argument("-n", "--num-vi-steps", nargs="?", default=5000, type=int)
    parser.add_argument('--num-acquisitions', nargs="?", default=10, type=int)
    parser.add_argument('--num-bo-steps', nargs="?", default=5, type=int)
    args = parser.parse_args()
    main(args.num_vi_steps, args.num_acquisitions, args.num_bo_steps)
