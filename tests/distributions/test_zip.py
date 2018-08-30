from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.distributions import ZIP, Poisson, Delta
from tests.common import assert_tensors_equal


@pytest.mark.parametrize('rate', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_0_gate(rate):
    # if gate is 0 ZIP is Poisson
    zip_ = ZIP(torch.zeros(1), torch.tensor(rate))
    pois = Poisson(torch.tensor(rate))
    s = pois.sample((20,))
    zip_prob = zip_.log_prob(s)
    pois_prob = pois.log_prob(s)
    assert_tensors_equal(zip_prob, pois_prob)


@pytest.mark.parametrize('rate', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_1_gate(rate):
    # if gate is 1 ZIP is Delta(0)
    zip_ = ZIP(torch.ones(1), torch.tensor(rate))
    delta = Delta(torch.zeros(1))
    s = torch.tensor([0., 1.])
    zip_prob = zip_.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_tensors_equal(zip_prob, delta_prob)
