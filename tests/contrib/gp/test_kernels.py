# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest
import torch

from pyro.contrib.gp.kernels import (RBF, Brownian, Constant, Coregionalize, Cosine, Exponent,
                                     Exponential, Linear, Matern32, Matern52, Periodic,
                                     Polynomial, Product, RationalQuadratic, Sum,
                                     VerticalScaling, Warping, WhiteNoise)
from tests.common import assert_equal

T = namedtuple("TestGPKernel", ["kernel", "X", "Z", "K_sum"])

variance = torch.tensor([3.0])
lengthscale = torch.tensor([2.0, 1.0, 2.0])
X = torch.tensor([[1.0, 0.0, 1.0], [2.0, 1.0, 3.0]])
Z = torch.tensor([[4.0, 5.0, 6.0], [3.0, 1.0, 7.0], [3.0, 1.0, 2.0]])

TEST_CASES = [
    T(
        Constant(3, variance),
        X=X, Z=Z, K_sum=18
    ),
    T(
        Brownian(1, variance),
        # only work on 1D input
        X=X[:, 0], Z=Z[:, 0], K_sum=27
    ),
    T(
        Cosine(3, variance, lengthscale),
        X=X, Z=Z, K_sum=-0.193233
    ),
    T(
        Linear(3, variance),
        X=X, Z=Z, K_sum=291
    ),
    T(
        Exponential(3, variance, lengthscale),
        X=X, Z=Z, K_sum=2.685679
    ),
    T(
        Matern32(3, variance, lengthscale),
        X=X, Z=Z, K_sum=3.229314
    ),
    T(
        Matern52(3, variance, lengthscale),
        X=X, Z=Z, K_sum=3.391847
    ),
    T(
        Periodic(3, variance, lengthscale, period=torch.ones(1)),
        X=X, Z=Z, K_sum=18
    ),
    T(
        Polynomial(3, variance, degree=2),
        X=X, Z=Z, K_sum=7017
    ),
    T(
        RationalQuadratic(3, variance, lengthscale, scale_mixture=torch.ones(1)),
        X=X, Z=Z, K_sum=5.684670
    ),
    T(
        RBF(3, variance, lengthscale),
        X=X, Z=Z, K_sum=3.681117
    ),
    T(
        WhiteNoise(3, variance, lengthscale),
        X=X, Z=Z, K_sum=0
    ),
    T(
        WhiteNoise(3, variance, lengthscale),
        X=X, Z=None, K_sum=6
    ),
    T(
        Coregionalize(3, components=torch.eye(3, 3)),
        X=torch.tensor([[1., 0., 0.],
                        [0.5, 0., 0.5]]),
        Z=torch.tensor([[1., 0., 0.],
                        [0., 1., 0.]]),
        K_sum=2.25,
    ),
    T(
        Coregionalize(3, rank=2),
        X=torch.tensor([[1., 0., 0.],
                        [0.5, 0., 0.5]]),
        Z=torch.tensor([[1., 0., 0.],
                        [0., 1., 0.]]),
        K_sum=None,  # kernel is randomly initialized
    ),
    T(
        Coregionalize(3),
        X=torch.tensor([[1., 0., 0.],
                        [0.5, 0., 0.5]]),
        Z=torch.tensor([[1., 0., 0.],
                        [0., 1., 0.]]),
        K_sum=None,  # kernel is randomly initialized
    ),
    T(
        Coregionalize(3, rank=2, diagonal=0.01 * torch.ones(3)),
        X=torch.tensor([[1., 0., 0.],
                        [0.5, 0., 0.5]]),
        Z=torch.tensor([[1., 0., 0.],
                        [0., 1., 0.]]),
        K_sum=None,  # kernel is randomly initialized
    ),
]

TEST_IDS = [t[0].__class__.__name__ for t in TEST_CASES]


@pytest.mark.parametrize("kernel, X, Z, K_sum", TEST_CASES, ids=TEST_IDS)
def test_kernel_forward(kernel, X, Z, K_sum):
    K = kernel(X, Z)
    assert K.shape == (X.shape[0], (X if Z is None else Z).shape[0])
    if K_sum is not None:
        assert_equal(K.sum().item(), K_sum)
    assert_equal(kernel(X).diag(), kernel(X, diag=True))
    if not isinstance(kernel, WhiteNoise):  # WhiteNoise avoids computing a delta function by assuming X != Z
        assert_equal(kernel(X), kernel(X, X))
    if Z is not None:
        assert_equal(kernel(X, Z), kernel(Z, X).t())


def test_combination():
    k0 = TEST_CASES[0][0]
    k5 = TEST_CASES[5][0]   # TEST_CASES[1] is Brownian, only work for 1D
    k2 = TEST_CASES[2][0]
    k3 = TEST_CASES[3][0]
    k4 = TEST_CASES[4][0]

    k = Sum(Product(Product(Sum(Sum(k0, k5), k2), 2), k3), Sum(k4, 1))

    K = 2 * (k0(X, Z) + k5(X, Z) + k2(X, Z)) * k3(X, Z) + (k4(X, Z) + 1)

    assert_equal(K.data, k(X, Z).data)


def test_active_dims_overlap_ok():
    k1 = Matern52(2, variance, lengthscale[0], active_dims=[0, 1])
    k2 = Matern32(2, variance, lengthscale[0], active_dims=[1, 2])
    Sum(k1, k2)


def test_active_dims_disjoint_ok():
    k1 = Matern52(2, variance, lengthscale[0], active_dims=[0, 1])
    k2 = Matern32(1, variance, lengthscale[0], active_dims=[2])
    Sum(k1, k2)


def test_transforming():
    k = TEST_CASES[6][0]

    def vscaling_fn(x):
        return x.sum(dim=1)

    def iwarping_fn(x):
        return x**2

    owarping_coef = [2, 0, 1, 3, 0]

    K = k(X, Z)
    K_iwarp = k(iwarping_fn(X), iwarping_fn(Z))
    K_owarp = 2 + K ** 2 + 3 * K ** 3
    K_vscale = vscaling_fn(X).unsqueeze(1) * K * vscaling_fn(Z).unsqueeze(0)

    assert_equal(K_iwarp.data, Warping(k, iwarping_fn=iwarping_fn)(X, Z).data)
    assert_equal(K_owarp.data, Warping(k, owarping_coef=owarping_coef)(X, Z).data)
    assert_equal(K_vscale.data, VerticalScaling(k, vscaling_fn=vscaling_fn)(X, Z).data)
    assert_equal(K.exp().data, Exponent(k)(X, Z).data)
