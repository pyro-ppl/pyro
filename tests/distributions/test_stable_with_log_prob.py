# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

import pyro
from pyro.distributions import StableWithLogProb as Stable
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from tests.common import assert_close

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize(
    "alpha, beta, c, mu",
    [
        (1.00, 0.8, 2.0, 3.0),
        (1.02, -0.8, 2.0, -3.0),
        (0.98, 0.5, 1.0, -3.0),
        (0.95, -0.5, 1.0, 3.0),
        (1.10, 0.0, 1.0, 0.0),
        (1.80, -0.5, 1.0, -2.0),
        (0.50, 0.0, 1.0, 2.0),
    ],
)
@pytest.mark.parametrize(
    "alpha_0, beta_0, c_0, mu_0",
    [
        (1.3, 0.0, 1.0, 0.0),
    ],
)
def test_stable_with_log_prob_param_fit(alpha, beta, c, mu, alpha_0, beta_0, c_0, mu_0):
    # Sample test data
    n = 10000
    pyro.set_rng_seed(20240520)
    data = Stable(alpha, beta, c, mu).sample((n,))

    def model(data):
        alpha = pyro.param(
            "alpha", torch.tensor(alpha_0), constraint=constraints.interval(0, 2)
        )
        beta = pyro.param(
            "beta", torch.tensor(beta_0), constraint=constraints.interval(-1, 1)
        )
        c = pyro.param("c", torch.tensor(c_0), constraint=constraints.positive)
        mu = pyro.param("mu", torch.tensor(mu_0), constraint=constraints.real)
        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", Stable(alpha, beta, c, mu), obs=data)

    def train(model, guide, num_steps=400, lr=0.03):
        pyro.clear_param_store()
        pyro.set_rng_seed(20240520)

        # set up ELBO, and optimizer
        elbo = Trace_ELBO()
        elbo.loss(model, guide, data=data)
        optim = pyro.optim.Adam({"lr": lr})
        svi = SVI(model, guide, optim, loss=elbo)

        # optimize
        for i in range(num_steps):
            loss = svi.step(data) / data.numel()
            if i % 10 == 0:
                logging.info(f"step {i} loss = {loss:0.6g}")
                log_progress()

        logging.info(f"Parameter estimates (n = {n}):")
        log_progress()

    def log_progress():
        logging.info(f"alpha: Estimate = {pyro.param('alpha')}, true = {alpha}")
        logging.info(f"beta: Estimate = {pyro.param('beta')}, true = {beta}")
        logging.info(f"c: Estimate = {pyro.param('c')}, true = {c}")
        logging.info(f"mu: Estimate = {pyro.param('mu')}, true = {mu}")

    # Fit model to data
    guide = AutoNormal(model)
    train(model, guide)

    # Verify fit accuracy
    assert_close(alpha, pyro.param("alpha").item(), atol=0.03)
    assert_close(beta, pyro.param("beta").item(), atol=0.06)
    assert_close(c, pyro.param("c").item(), atol=0.2)
    assert_close(mu, pyro.param("mu").item(), atol=0.2)


# # The below tests will be executed:
# test_stable_with_log_prob_param_fit(1.00,  0.8,  2.0,  3.0,  1.3,  0.0,  1.0,  0.0)
# test_stable_with_log_prob_param_fit(1.02, -0.8,  2.0, -3.0,  1.3,  0.0,  1.0,  0.0)
# test_stable_with_log_prob_param_fit(0.98,  0.5,  1.0, -3.0,  1.3,  0.0,  1.0,  0.0)
# test_stable_with_log_prob_param_fit(0.95, -0.5,  1.0,  3.0,  1.3,  0.0,  1.0,  0.0)
# test_stable_with_log_prob_param_fit(1.10,  0.0,  1.0,  0.0,  1.3,  0.0,  1.0,  0.0)
# test_stable_with_log_prob_param_fit(1.80, -0.5,  1.0, -2.0,  1.3,  0.0,  1.0,  0.0)
# test_stable_with_log_prob_param_fit(0.50,  0.0,  1.0,  2.0,  1.3,  0.0,  1.0,  0.0)
