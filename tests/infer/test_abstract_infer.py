from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc import MCMC, NUTS


def model(num_trials):
    phi_prior = dist.Uniform(num_trials.new_tensor(0.), num_trials.new_tensor(1.))\
        .expand_by(num_trials.shape[0])
    success_prob = pyro.sample("phi", phi_prior)
    return pyro.sample("obs", dist.Binomial(success_prob))


def conditioned_model(model, num_trials, num_success):
    return poutine.condition(model, data={"obs": num_success})(num_trials)


def test_posterior_predictive():
    data = torch.ones(5) * 0.7
    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc_posterior = MCMC(nuts_kernel)