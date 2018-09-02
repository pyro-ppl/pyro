from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.distributions import ZeroInflatedPoisson, Poisson, Delta
from tests.common import assert_tensors_equal


@pytest.mark.parametrize('rate', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_0_gate(rate):
    # if gate is 0 ZIP is Poisson
    zip_ = ZeroInflatedPoisson(torch.zeros(1), torch.tensor(rate))
    pois = Poisson(torch.tensor(rate))
    s = pois.sample((20,))
    zip_prob = zip_.log_prob(s)
    pois_prob = pois.log_prob(s)
    assert_tensors_equal(zip_prob, pois_prob)


@pytest.mark.parametrize('rate', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_1_gate(rate):
    # if gate is 1 ZIP is Delta(0)
    zip_ = ZeroInflatedPoisson(torch.ones(1), torch.tensor(rate))
    delta = Delta(torch.zeros(1))
    s = torch.tensor([0., 1.])
    zip_prob = zip_.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_tensors_equal(zip_prob, delta_prob)


@pytest.mark.parametrize('gate', [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize('rate', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_mean_variance(gate, rate):
    num_samples = 1000000
    zip_ = ZeroInflatedPoisson(torch.tensor(gate), torch.tensor(rate))
    s = zip_.sample((num_samples, ))
    expected_mean = zip_.mean
    estimated_mean = s.mean()
    expected_std = zip_.stddev
    estimated_std = s.std()
    assert_tensors_equal(expected_mean, estimated_mean, prec=1e-02)
    assert_tensors_equal(expected_std, estimated_std, prec=1e-02)
