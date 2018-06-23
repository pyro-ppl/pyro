from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import EmpiricalMarginal, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
from tests.common import assert_equal


def model(num_trials):
    phi_prior = dist.Uniform(num_trials.new_tensor(0.), num_trials.new_tensor(1.))\
        .expand_by([num_trials.shape[0]])
    success_prob = pyro.sample("phi", phi_prior)
    return pyro.sample("obs", dist.Binomial(num_trials, success_prob))


def test_posterior_predictive():
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    nuts_kernel = NUTS(conditioned_model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200).run(num_trials)
    posterior_predictive = TracePredictive(model, mcmc_run, num_samples=10000).run(num_trials)
    marginal_return_vals = EmpiricalMarginal(posterior_predictive)
    assert_equal(marginal_return_vals.mean, torch.ones(5) * 700, prec=30)


def test_nesting():
    def nested():
        true_probs = torch.ones(5) * 0.7
        num_trials = torch.ones(5) * 1000
        num_success = dist.Binomial(num_trials, true_probs).sample()
        conditioned_model = poutine.condition(model, data={"obs": num_success})
        nuts_kernel = NUTS(conditioned_model, adapt_step_size=True)
        mcmc_run = MCMC(nuts_kernel, num_samples=10, warmup_steps=2).run(num_trials)
        return mcmc_run

    with poutine.trace() as tp:
        nested()
        nested()

    assert len(tp.trace.nodes) == 0
