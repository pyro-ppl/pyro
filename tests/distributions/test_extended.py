# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_equal


def test_extended_binomial():
    total_count = torch.tensor([1., 2., 10.])
    probs = torch.tensor([0.5, 0.4, 0.2])

    d1 = dist.Binomial(total_count, probs)
    d2 = dist.ExtendedBinomial(total_count, probs)

    # Check on good data.
    data = d1.sample((100,))
    assert_equal(d1.log_prob(data), d2.log_prob(data))

    # Check on extended data.
    data = torch.arange(-10., 20.).unsqueeze(-1)
    with pytest.raises(ValueError):
        d1.log_prob(data)
    log_prob = d2.log_prob(data)
    valid = d1.support.check(data)
    assert ((log_prob > -math.inf) == valid).all()

    # Check on shape error.
    with pytest.raises(ValueError):
        d2.log_prob(torch.tensor([0., 0.]))

    # Check on value error.
    with pytest.raises(ValueError):
        d2.log_prob(torch.tensor(0.5))

    # Check on negative total_count.
    total_count = torch.arange(-10, 0.)
    d = dist.ExtendedBinomial(total_count, 0.5)
    assert (d.log_prob(data) == -math.inf).all()


def test_extended_beta_binomial():
    concentration1 = torch.tensor([1.0, 2.0, 1.0])
    concentration0 = torch.tensor([0.5, 1.0, 2.0])
    total_count = torch.tensor([1., 2., 10.])

    d1 = dist.BetaBinomial(concentration1, concentration0, total_count)
    d2 = dist.ExtendedBetaBinomial(concentration1, concentration0, total_count)

    # Check on good data.
    data = d1.sample((100,))
    assert_equal(d1.log_prob(data), d2.log_prob(data))

    # Check on extended data.
    data = torch.arange(-10., 20.).unsqueeze(-1)
    with pytest.raises(ValueError):
        d1.log_prob(data)
    log_prob = d2.log_prob(data)
    valid = d1.support.check(data)
    assert ((log_prob > -math.inf) == valid).all()

    # Check on shape error.
    with pytest.raises(ValueError):
        d2.log_prob(torch.tensor([0., 0.]))

    # Check on value error.
    with pytest.raises(ValueError):
        d2.log_prob(torch.tensor(0.5))

    # Check on negative total_count.
    total_count = torch.arange(-10, 0.)
    d = dist.ExtendedBetaBinomial(1.5, 1.5, total_count)
    assert (d.log_prob(data) == -math.inf).all()
