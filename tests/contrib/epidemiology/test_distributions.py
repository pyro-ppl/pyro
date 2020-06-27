# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.distributions.transforms import SigmoidTransform

import pyro.distributions as dist
from pyro.contrib.epidemiology import beta_binomial_dist, binomial_dist, infection_dist
from pyro.contrib.epidemiology.distributions import _RELAX_MIN_VARIANCE, set_relaxed_distributions
from tests.common import assert_close


def assert_dist_close(d1, d2):
    x = torch.arange(float(200))
    p1 = d1.log_prob(x).exp()
    p2 = d2.log_prob(x).exp()

    assert (p1.sum() - 1).abs() < 1e-3, "incomplete mass"
    assert (p2.sum() - 1).abs() < 1e-3, "incomplete mass"

    mean1 = (p1 * x).sum()
    mean2 = (p2 * x).sum()
    assert_close(mean1, mean2, rtol=0.05)

    max_prob = torch.max(p1.max(), p2.max())
    assert (p1 - p2).abs().max() / max_prob < 0.05


@pytest.mark.parametrize("R0,I", [
    (1., 1),
    (1., 10),
    (10., 1),
    (5., 5),
])
def test_binomial_vs_poisson(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=1000., population=1000.)

    assert isinstance(d1, dist.Poisson)
    assert isinstance(d2, dist.Binomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I,k", [
    (1., 1., 0.5),
    (1., 1., 1.),
    (1., 1., 2.),
    (1., 10., 0.5),
    (1., 10., 1.),
    (1., 10., 2.),
    (10., 1., 0.5),
    (10., 1., 1.),
    (10., 1., 2.),
    (5., 5, 0.5),
    (5., 5, 1.),
    (5., 5, 2.),
])
def test_beta_binomial_vs_negative_binomial(R0, I, k):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I, concentration=k)
    d2 = infection_dist(individual_rate=R0, num_infectious=I, concentration=k,
                        num_susceptible=1000., population=1000.)

    assert isinstance(d1, dist.NegativeBinomial)
    assert isinstance(d2, dist.BetaBinomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I", [
    (1., 1.),
    (1., 10.),
    (10., 1.),
    (5., 5.),
])
def test_beta_binomial_vs_binomial(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=20., population=30.)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=20., population=30.,
                        concentration=200.)

    assert isinstance(d1, dist.Binomial)
    assert isinstance(d2, dist.BetaBinomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I", [
    (1., 1.),
    (1., 10.),
    (10., 1.),
    (5., 5.),
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


@pytest.mark.parametrize("overdispersion", [0.01, 0.03, 0.1, 0.3, 1.0, 1.5])
@pytest.mark.parametrize("probs", [0.01, 0.03, 0.1, 0.3, 0.7, 0.9, 0.97, 0.99])
def test_overdispersed_bound(probs, overdispersion):
    total_count = torch.tensor([1, 2, 5, 10, 20, 50, 1e2, 1e3, 1e5, 1e6, 1e7, 1e8])
    d = binomial_dist(total_count, probs, overdispersion=overdispersion)
    relative_error = d.variance.sqrt() / (probs * (1 - probs) * total_count)

    # Check bound is valid.
    assert (relative_error >= overdispersion).all()

    # Check bound is tight.
    assert relative_error[-1] / overdispersion < 1.05


@pytest.mark.parametrize("overdispersion", [0.05, 0.1, 0.2, 0.3])
@pytest.mark.parametrize("probs", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_overdispersed_asymptote(probs, overdispersion):
    total_count = 100000

    # Check binomial_dist converges in distribution to LogitNormal.
    d1 = binomial_dist(total_count, probs)
    d2 = dist.TransformedDistribution(
        dist.Normal(math.log(probs / (1 - probs)), overdispersion),
        SigmoidTransform())

    # CRPS is equivalent to the Cramer-von Mises test.
    # https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion
    k = torch.arange(0., total_count + 1.)
    cdf1 = d1.log_prob(k).exp().cumsum(-1)
    cdf2 = d2.cdf(k / total_count)
    crps = (cdf1 - cdf2).pow(2).mean()
    assert crps < 0.02


@pytest.mark.parametrize("total_count", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("concentration1", [0.2, 1.0, 5.])
@pytest.mark.parametrize("concentration0", [0.2, 1.0, 5.])
def test_beta_binomial(concentration1, concentration0, total_count):
    # For small overdispersion, beta_binomial_dist is close to BetaBinomial.
    d1 = dist.BetaBinomial(concentration1, concentration0, total_count)
    d2 = beta_binomial_dist(concentration1, concentration0, total_count,
                            overdispersion=0.01)

    # CRPS is equivalent to the Cramer-von Mises test.
    # https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion
    k = torch.arange(0., total_count + 1.)
    cdf1 = d1.log_prob(k).exp().cumsum(-1)
    cdf2 = d2.log_prob(k).exp().cumsum(-1)
    crps = (cdf1 - cdf2).pow(2).mean()
    assert crps < 0.01


@pytest.mark.parametrize("overdispersion", [0.05, 0.1, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("total_count", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("probs", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_overdispersed_beta_binomial(probs, total_count, overdispersion):
    # For high concentraion, beta_binomial_dist is close to binomial_dist.
    concentration = 100.  # very little uncertainty
    concentration1 = concentration * probs
    concentration0 = concentration * (1 - probs)
    d1 = binomial_dist(total_count, probs, overdispersion=overdispersion)
    d2 = beta_binomial_dist(concentration1, concentration0, total_count,
                            overdispersion=overdispersion)

    # CRPS is equivalent to the Cramer-von Mises test.
    # https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion
    k = torch.arange(0., total_count + 1.)
    cdf1 = d1.log_prob(k).exp().cumsum(-1)
    cdf2 = d2.log_prob(k).exp().cumsum(-1)
    crps = (cdf1 - cdf2).pow(2).mean()
    assert crps < 0.01


def test_relaxed_binomial():
    total_count = torch.arange(1, 33)
    probs = torch.linspace(0.1, 0.9, 16).unsqueeze(-1)

    d1 = binomial_dist(total_count, probs)
    assert isinstance(d1, dist.ExtendedBinomial)

    with set_relaxed_distributions():
        d2 = binomial_dist(total_count, probs)
    assert isinstance(d2, dist.Normal)
    assert_close(d2.mean, d1.mean)
    assert_close(d2.variance, d1.variance.clamp(min=_RELAX_MIN_VARIANCE))


@pytest.mark.parametrize("overdispersion", [0.05, 0.1, 0.2, 0.5, 1.0])
def test_relaxed_overdispersed_binomial(overdispersion):
    total_count = torch.arange(1, 33)
    probs = torch.linspace(0.1, 0.9, 16).unsqueeze(-1)

    d1 = binomial_dist(total_count, probs, overdispersion=overdispersion)
    assert isinstance(d1, dist.ExtendedBetaBinomial)

    with set_relaxed_distributions():
        d2 = binomial_dist(total_count, probs, overdispersion=overdispersion)
    assert isinstance(d2, dist.Normal)
    assert_close(d2.mean, d1.mean)
    assert_close(d2.variance, d1.variance.clamp(min=_RELAX_MIN_VARIANCE))


def test_relaxed_beta_binomial():
    total_count = torch.arange(1, 17)
    concentration1 = torch.logspace(-1, 2, 8).unsqueeze(-1)
    concentration0 = concentration1.unsqueeze(-1)

    d1 = beta_binomial_dist(concentration1, concentration0, total_count)
    assert isinstance(d1, dist.ExtendedBetaBinomial)

    with set_relaxed_distributions():
        d2 = beta_binomial_dist(concentration1, concentration0, total_count)
    assert isinstance(d2, dist.Normal)
    assert_close(d2.mean, d1.mean)
    assert_close(d2.variance, d1.variance.clamp(min=_RELAX_MIN_VARIANCE))


@pytest.mark.parametrize("overdispersion", [0.05, 0.1, 0.2, 0.5, 1.0])
def test_relaxed_overdispersed_beta_binomial(overdispersion):
    total_count = torch.arange(1, 17)
    concentration1 = torch.logspace(-1, 2, 8).unsqueeze(-1)
    concentration0 = concentration1.unsqueeze(-1)

    d1 = beta_binomial_dist(concentration1, concentration0, total_count,
                            overdispersion=overdispersion)
    assert isinstance(d1, dist.ExtendedBetaBinomial)

    with set_relaxed_distributions():
        d2 = beta_binomial_dist(concentration1, concentration0, total_count,
                                overdispersion=overdispersion)
    assert isinstance(d2, dist.Normal)
    assert_close(d2.mean, d1.mean)
    assert_close(d2.variance, d1.variance.clamp(min=_RELAX_MIN_VARIANCE))
