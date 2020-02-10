# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch
from torch import optim

from pyro.distributions import VonMises, VonMises3D
from pyro.distributions.von_mises import _log_modified_bessel_fn
from tests.common import skipif_param


def _fit_params_from_samples(samples, n_iter):
    assert samples.dim() == 1
    samples_count = samples.size(0)
    samples_cs = samples.cos().sum()
    samples_ss = samples.sin().sum()
    mu = torch.atan2(samples_ss / samples_count, samples_cs / samples_count)
    samples_r = (samples_cs ** 2 + samples_ss ** 2).sqrt() / samples_count
    # From Banerjee, Arindam, et al.
    # "Clustering on the unit hypersphere using von Mises-Fisher distributions."
    # Journal of Machine Learning Research 6.Sep (2005): 1345-1382.
    # By mic (https://stats.stackexchange.com/users/67168/mic),
    # Estimating kappa of von Mises distribution, URL (version: 2015-06-12):
    # https://stats.stackexchange.com/q/156692
    kappa = (samples_r * 2 - samples_r ** 3) / (1 - samples_r ** 2)
    lr = 1e-2
    kappa.requires_grad = True
    bfgs = optim.LBFGS([kappa], lr=lr)

    def bfgs_closure():
        bfgs.zero_grad()
        obj = (_log_modified_bessel_fn(kappa, order=1)
               - _log_modified_bessel_fn(kappa, order=0))
        obj = (obj - samples_r.log()).abs()
        obj.backward()
        return obj

    for i in range(n_iter):
        bfgs.step(bfgs_closure)
    return mu, kappa.detach()


@pytest.mark.parametrize('loc', [-math.pi/2.0, 0.0, math.pi/2.0])
@pytest.mark.parametrize('concentration', [skipif_param(0.01, condition='CUDA_TEST' in os.environ,
                                                        reason='low precision.'),
                                           0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
def test_sample(loc, concentration, n_samples=int(1e6), n_iter=50):
    prob = VonMises(loc, concentration)
    samples = prob.sample((n_samples,))
    mu, kappa = _fit_params_from_samples(samples, n_iter=n_iter)
    assert abs(loc - mu) < 0.1
    assert abs(concentration - kappa) < concentration * 0.1


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
