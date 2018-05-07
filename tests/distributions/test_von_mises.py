from __future__ import absolute_import, division, print_function

import math

import pytest
import torch

from pyro.distributions import VonMises


@pytest.mark.parametrize('concentration', [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
def test_log_prob_normalized(concentration):
    grid = torch.arange(0, 2 * math.pi, 1e-4)
    prob = VonMises(0.0, concentration).log_prob(grid).exp()
    norm = prob.mean().item() * 2 * math.pi
    assert abs(norm - 1) < 1e-3, norm
