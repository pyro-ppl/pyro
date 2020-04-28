# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.epidemiology import infection_dist

from tests.common import assert_close


def assert_dist_close(d1, d2):
    x = torch.arange(float(200))
    p1 = d1.log_prob(x).exp()
    p2 = d2.log_prob(x).exp()

    assert (p1.sum() - 1).abs() < 1e-3, "incomplete mass"
    assert (p2.sum() - 1).abs() < 1e-3, "incomplete mass"

    mean1 = (p1 * x).sum()
    mean2 = (p2 * x).sum()
    assert_close(mean1, mean2)

    max_prob = torch.max(p1.max(), p2.max())
    assert (p1 - p2).abs().max() / max_prob < 0.05


@pytest.mark.parametrize("R0,I", [
    (1.0, 1),
    (1.0, 10),
    (10.0, 1),
    (10.0, 10),
])
def test_binomial_vs_poisson(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=1000, population=1000)

    assert isinstance(d1, dist.Poisson)
    assert isinstance(d2, dist.Binomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I,k", [
    (1.0, 1, 0.5),
    (1.0, 10, 0.5),
    (10.0, 1, 0.5),
    (10.0, 10, 0.5),
    (1.0, 1, 1.0),
    (1.0, 10, 1.0),
    (10.0, 1, 1.0),
    (10.0, 10, 1.0),
    (1.0, 1, 2.0),
    (1.0, 10, 2.0),
    (10.0, 1, 2.0),
    (10.0, 10, 2.0),
])
def test_beta_binomial_vs_negative_binomial(R0, I, k):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I, concentration=k)
    d2 = infection_dist(individual_rate=R0, num_infectious=I, concentration=k,
                        num_susceptible=1000, population=1000)

    assert isinstance(d1, dist.NegativeBinomial)
    assert isinstance(d2, dist.BetaBinomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I", [
    (1.0, 1),
    (1.0, 10),
    (10.0, 1),
    (10.0, 10),
])
def test_beta_binomial_vs_binomial(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=20, population=30)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=20, population=30,
                        concentration=200.)

    assert isinstance(d1, dist.Binomial)
    assert isinstance(d2, dist.BetaBinomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I", [
    (1.0, 1),
    (1.0, 10),
    (10.0, 1),
    (10.0, 10),
])
def test_negative_binomial_vs_poisson(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        concentration=200.)

    assert isinstance(d1, dist.Poisson)
    assert isinstance(d2, dist.NegativeBinomial)
    assert_dist_close(d1, d2)
