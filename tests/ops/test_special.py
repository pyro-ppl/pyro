# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from scipy.special import iv
from torch import tensor
from torch.autograd import grad

from pyro.ops.special import get_quad_rule, log_beta, log_binomial, log_I1, safe_log
from tests.common import assert_equal


def test_safe_log():
    # Test values.
    x = torch.randn(1000).exp().requires_grad_()
    expected = x.log()
    actual = safe_log(x)
    assert_equal(actual, expected)
    assert_equal(grad(actual.sum(), [x])[0], grad(expected.sum(), [x])[0])

    # Test gradients.
    x = torch.tensor(0.0, requires_grad=True)
    assert not torch.isfinite(grad(x.log(), [x])[0])
    assert torch.isfinite(grad(safe_log(x), [x])[0])


@pytest.mark.parametrize(
    "tol",
    [
        1e-8,
        1e-6,
        1e-4,
        1e-2,
        0.02,
        0.05,
        0.1,
        0.2,
        0.1,
        1.0,
    ],
)
def test_log_beta_stirling(tol):
    x = torch.logspace(-5, 5, 200)
    y = x.unsqueeze(-1)

    expected = log_beta(x, y)
    actual = log_beta(x, y, tol=tol)

    assert (actual <= expected).all()
    assert (expected < actual + tol).all()


@pytest.mark.parametrize(
    "tol",
    [
        1e-8,
        1e-6,
        1e-4,
        1e-2,
        0.02,
        0.05,
        0.1,
        0.2,
        0.1,
        1.0,
    ],
)
def test_log_binomial_stirling(tol):
    k = torch.arange(200.0)
    n_minus_k = k.unsqueeze(-1)
    n = k + n_minus_k

    # Test binomial coefficient choose(n, k).
    expected = (n + 1).lgamma() - (k + 1).lgamma() - (n_minus_k + 1).lgamma()
    actual = log_binomial(n, k, tol=tol)

    assert (actual - expected).abs().max() < tol


@pytest.mark.parametrize("order", [0, 1, 5, 10, 20])
@pytest.mark.parametrize("value", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_log_I1(order, value):
    value = tensor([value])
    expected = torch.tensor([iv(i, value.numpy()) for i in range(order + 1)]).log()
    actual = log_I1(order, value)
    assert_equal(actual, expected)


def test_log_I1_shapes():
    assert_equal(log_I1(10, tensor(0.6)).shape, torch.Size([11, 1]))
    assert_equal(log_I1(10, tensor([0.6])).shape, torch.Size([11, 1]))
    assert_equal(log_I1(10, tensor([[0.6]])).shape, torch.Size([11, 1, 1]))
    assert_equal(log_I1(10, tensor([0.6, 0.2])).shape, torch.Size([11, 2]))
    assert_equal(log_I1(0, tensor(0.6)).shape, torch.Size((1, 1)))


@pytest.mark.parametrize("sigma", [0.5, 1.25])
def test_get_quad_rule(sigma):
    quad_points, log_weights = get_quad_rule(32, torch.zeros(1))
    quad_points *= sigma  # transform to N(0, sigma) gaussian
    variance = torch.logsumexp(quad_points.pow(2.0).log() + log_weights, axis=0).exp()
    assert_equal(sigma**2, variance.item())
