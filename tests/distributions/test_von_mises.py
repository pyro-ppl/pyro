# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch
from torch import optim

from pyro.distributions import VonMises, VonMises3D
from tests.common import skipif_param


def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                  -0.2057706e-1, 0.2635537e-1, -0.1647633e-1,  0.392377e-2]
_I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3]
_I1_COEF_LARGE = [0.39894228, -0.3988024e-1, -0.362018e-2, 0.163801e-2, -0.1031555e-1,
                  0.2282967e-1, -0.2895312e-1, 0.1787654e-1, -0.420059e-2]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    """
    Returns ``log(I_order(x))`` for ``x > 0``,
    where `order` is either 0 or 1.
    """
    assert order == 0 or order == 1

    # compute small solution
    y = (x / 3.75).pow(2)
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    mask = (x < 3.75)
    result = large
    if mask.any():
        result[mask] = small[mask]
    return result


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
