# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
from scipy.stats import wasserstein_distance

import pyro.distributions as dist
import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("n", [2, 3, 5, 10])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test_binomial_approximation(n, p):
    num_samples = 10000
    d1 = dist.Binomial(n, p)
    d2 = dist.DequantizedDistribution(d1)

    x1 = d1.sample((num_samples,))
    x2 = d2.sample((num_samples,))

    error = wasserstein_distance(x1, x2)
    logger.debug("error = {:0.3g}".format(error))
    assert error < 1/3


@pytest.mark.parametrize("rate", [0.1, 1.0, 2.0, 10.0])
def test_poisson_approximation(rate):
    num_samples = 10000
    d1 = dist.Poisson(rate)
    d2 = dist.DequantizedDistribution(d1)

    x1 = d1.sample((num_samples,))
    x2 = d2.sample((num_samples,))

    error = wasserstein_distance(x1, x2)
    logger.debug("error = {:0.3g}".format(error))
    assert error < 1/3
