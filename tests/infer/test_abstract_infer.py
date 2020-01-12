# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoLaplaceApproximation
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from tests.common import assert_equal


pytestmark = pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")


def model(num_trials):
    with pyro.plate("data", num_trials.size(0)):
        phi_prior = dist.Uniform(num_trials.new_tensor(0.), num_trials.new_tensor(1.))
        success_prob = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", dist.Binomial(num_trials, success_prob))


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


# TODO: Make this available directly in `SVI` if needed.
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_information_criterion():
    # milk dataset: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
    kcal = torch.tensor([0.49, 0.47, 0.56, 0.89, 0.92, 0.8, 0.46, 0.71, 0.68,
                         0.97, 0.84, 0.62, 0.54, 0.49, 0.48, 0.55, 0.71])
    kcal_mean = kcal.mean()
    kcal_logstd = kcal.std().log()

    def model():
        mu = pyro.sample("mu", dist.Normal(kcal_mean, 1))
        log_sigma = pyro.sample("log_sigma", dist.Normal(kcal_logstd, 1))
        with pyro.plate("plate"):
            pyro.sample("kcal", dist.Normal(mu, log_sigma.exp()), obs=kcal)

    delta_guide = AutoLaplaceApproximation(model)

    svi = SVI(model, delta_guide, optim.Adam({"lr": 0.05}), loss=Trace_ELBO(),
              num_steps=0, num_samples=3000)
    for i in range(100):
        svi.step()

    svi.guide = delta_guide.laplace_approximation()
    posterior = svi.run()

    ic = posterior.information_criterion()
    assert_equal(ic["waic"], torch.tensor(-8.3), prec=0.2)
    assert_equal(ic["p_waic"], torch.tensor(1.8), prec=0.2)
