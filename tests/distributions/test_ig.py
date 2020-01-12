# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import pytest
from pyro.distributions import Gamma, InverseGamma
from tests.common import assert_equal


@pytest.mark.parametrize('concentration', [3.3, 4.0])
@pytest.mark.parametrize('rate', [2.5, 3.0])
def test_sample(concentration, rate, n_samples=int(1e6)):
    samples = InverseGamma(concentration, rate).sample((n_samples,))
    mean, std = samples.mean().item(), samples.std().item()
    expected_mean = rate / (concentration - 1.0)
    expected_std = rate / ((concentration - 1.0) * math.sqrt(concentration - 2.0))
    assert_equal(mean, expected_mean, prec=1e-2)
    assert_equal(std, expected_std, prec=0.03)


@pytest.mark.parametrize('concentration', [2.5, 4.0])
@pytest.mark.parametrize('rate', [2.5, 3.0])
@pytest.mark.parametrize('value', [0.5, 1.7])
def test_log_prob(concentration, rate, value):
    value = torch.tensor(value)
    log_prob = InverseGamma(concentration, rate).log_prob(value)
    expected_log_prob = Gamma(concentration, rate).log_prob(1.0 / value) - 2.0 * value.log()
    assert_equal(log_prob, expected_log_prob, prec=1e-6)
