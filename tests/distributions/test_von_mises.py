from __future__ import absolute_import, division, print_function

import math

import pytest
import torch

from pyro.distributions import VonMises, VonMises3D


@pytest.mark.parametrize('concentration', [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
def test_log_prob_normalized(concentration):
    grid = torch.arange(0., 2 * math.pi, 1e-4)
    prob = VonMises(0.0, concentration).log_prob(grid).exp()
    norm = prob.mean().item() * 2 * math.pi
    assert abs(norm - 1) < 1e-3, norm


@pytest.mark.parametrize('scale', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_von_mises_3d(scale):
    concentration = torch.randn(3)
    concentration = concentration * (scale / concentration.norm(2))

    num_samples = 100000
    samples = torch.randn(num_samples, 3)
    samples = samples / samples.norm(2, dim=-1, keepdim=True)

    d = VonMises3D(concentration, validate_args=True)
    actual_total = d.log_prob(samples).exp().mean()
    expected_total = 1 / (4 * math.pi)
    ratio = actual_total / expected_total
    assert torch.abs(ratio - 1) < 0.01, ratio
