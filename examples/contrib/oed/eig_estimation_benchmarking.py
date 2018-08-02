import argparse
from functools import partial
import torch
from torch.distributions import constraints

import pyro
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_ape

from models.bayes_linear import (
    bayesian_linear_model, normal_inv_gamma_guide, group_assignment_matrix,
    analytic_posterior_entropy
)

# Set up regression model dimensions
N = 100  # number of participants
p = 2    # number of features
prior_sds = torch.tensor([10., 2.5])

# Model and guide using known obs_sd
model = partial(bayesian_linear_model, w_mean=torch.tensor(0.),
                w_sqrtlambda=1/prior_sds, obs_sd=torch.tensor(1.))
guide = partial(normal_inv_gamma_guide, obs_sd=torch.tensor(1.))


def estimated_ape(ns, model, guide, num_vi_steps):
    designs = [group_assignment_matrix(torch.tensor([n1, N-n1])) for n1 in ns]
    X = torch.stack(designs)
    est_ape = vi_ape(
        model,
        X,
        observation_labels="y",
        vi_parameters={
            "guide": guide,
            "optim": optim.Adam({"lr": 0.0025}),
            "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
            "num_steps": num_vi_steps},
        is_parameters={"num_samples": 1}
    )
    return est_ape


def true_ape(ns, model, guide):
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

    # todo timer
    for f in [true_ape, est_ape]:
        X = torch.tensor(range(0, N, 5))
        y = f(X)
        print(y)
        print(timer)


if __name__ == "__main__":
    # todo change
    parser = argparse.ArgumentParser(description="A/B test experiment design using VI")
    parser.add_argument("-n", "--num-vi-steps", nargs="?", default=5000, type=int)
    parser.add_argument('--num-acquisitions', nargs="?", default=10, type=int)
    parser.add_argument('--num-bo-steps', nargs="?", default=6, type=int)
    args = parser.parse_args()
    main(args.num_vi_steps, args.num_acquisitions, args.num_bo_steps)
