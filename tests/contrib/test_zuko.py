# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import pyro
from pyro.contrib.zuko import ZukoToPyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("rsample_and_log_prob", [True, False])
def test_ZukoToPyro(multivariate: bool, rsample_and_log_prob: bool):
    # Distribution
    if multivariate:
        normal = torch.distributions.MultivariateNormal
        mu = torch.zeros(3)
        sigma = torch.eye(3)
    else:
        normal = torch.distributions.Normal
        mu = torch.zeros(())
        sigma = torch.ones(())

    dist = normal(mu, sigma)

    if rsample_and_log_prob:

        def dummy(self, shape):
            x = self.rsample(shape)
            return x, self.log_prob(x)

        dist.rsample_and_log_prob = dummy.__get__(dist)

    # Sample
    x1 = pyro.sample("x1", ZukoToPyro(dist))

    assert x1.shape == dist.event_shape

    # Sample within plate
    with pyro.plate("data", 4):
        x2 = pyro.sample("x2", ZukoToPyro(dist))

    assert x2.shape == (4, *dist.event_shape)

    # SVI
    def model():
        pyro.sample("a", ZukoToPyro(dist))

        with pyro.plate("data", 4):
            pyro.sample("b", ZukoToPyro(dist))

    def guide():
        mu_ = pyro.param("mu", mu)
        sigma_ = pyro.param("sigma", sigma)

        pyro.sample("a", ZukoToPyro(normal(mu_, sigma_)))

        with pyro.plate("data", 4):
            pyro.sample("b", ZukoToPyro(normal(mu_, sigma_)))

    svi = SVI(model, guide, optim=Adam({"lr": 1e-3}), loss=Trace_ELBO())
    svi.step()
