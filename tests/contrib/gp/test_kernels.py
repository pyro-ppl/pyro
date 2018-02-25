from __future__ import absolute_import, division, print_function

from collections import namedtuple

import pytest
import torch
from torch.autograd import Variable

from pyro.contrib.gp.kernels import (Bias, Brownian, Cosine, Linear, Matern12, Matern32,
                                     Matern52, Periodic, Polynomial, RationalQuadratic,
                                     SquaredExponential, WhiteNoise)
from tests.common import assert_equal

T = namedtuple("TestKernelForward", ["kernel", "X", "Z", "K_sum"])

variance = torch.Tensor([3])
lengthscale = torch.Tensor([2, 1, 2])
X = Variable(torch.Tensor([[1, 0, 1], [2, 1, 3]]))
Z = Variable(torch.Tensor([[4, 5, 6], [3, 1, 7], [3, 1, 2]]))

TEST_CASES = [
    T(
        Bias(3, variance),
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
        Matern12(3, variance, lengthscale),
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
        SquaredExponential(3, variance, lengthscale),
        X=X, Z=Z, K_sum=3.681117
    ),
    T(
        WhiteNoise(3, variance, lengthscale),
        X=X, Z=Z, K_sum=0
    ),
    T(
        WhiteNoise(3, variance, lengthscale),
        X=X, Z=None, K_sum=6
    )
]

TEST_IDS = [t[0].__class__.__name__ for t in TEST_CASES]


@pytest.mark.parametrize("kernel, X, Z, K_sum", TEST_CASES, ids=TEST_IDS)
def test_kernel_forward(kernel, X, Z, K_sum):
    K = kernel(X, Z)
    assert K.dim() == 2
    assert K.size(0) == 2
    assert K.size(1) == (3 if Z is not None else 2)
    assert_equal(K.data.sum(), K_sum)
    assert_equal(kernel(X).diag(), kernel(X, diag=True))
