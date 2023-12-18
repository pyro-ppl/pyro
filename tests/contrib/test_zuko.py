# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import pytest
import pyro
import torch

from pyro.contrib.zuko import Zuko2Pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


@pytest.mark.parametrize("multivariate", [True, False])
def test_Zuko2Pyro(multivariate: bool):
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

    # Sample
    x1 = pyro.sample("x1", Zuko2Pyro(dist))

    assert x1.shape == dist.event_shape

    # Sample within plate
    with pyro.plate("data", 4):
        x2 = pyro.sample("x2", Zuko2Pyro(dist))

    assert x2.shape == (4, *dist.event_shape)

    # SVI
    def model():
        pyro.sample("a", Zuko2Pyro(dist))

        with pyro.plate("data", 4):
            pyro.sample("b", Zuko2Pyro(dist))

    def guide():
        mu_ = pyro.param("mu", mu)
        sigma_ = pyro.param("sigma", sigma)

        pyro.sample("a", Zuko2Pyro(normal(mu_, sigma_)))

        with pyro.plate("data", 4):
            pyro.sample("b", Zuko2Pyro(normal(mu_, sigma_)))

    svi = SVI(model, guide, optim=Adam({"lr": 1e-3}), loss=Trace_ELBO())
    svi.step()
