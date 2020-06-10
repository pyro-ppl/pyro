# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

from pyro.contrib.epidemiology.util import cat2, clamp, quantize_enumerate
from tests.common import assert_close, assert_equal


@pytest.mark.parametrize("min", [None, 0., (), (2,)], ids=str)
@pytest.mark.parametrize("max", [None, 1., (), (2,)], ids=str)
@pytest.mark.parametrize("shape", [(2,), (3, 2)], ids=str)
def test_clamp(shape, min, max):
    tensor = torch.randn(shape)
    if isinstance(min, tuple):
        min = torch.zeros(min)
    if isinstance(max, tuple):
        max = torch.ones(max)

    actual = clamp(tensor, min=min, max=max)

    expected = tensor
    if min is not None:
        min = torch.as_tensor(min).expand_as(tensor)
        expected = torch.max(min, expected)
    if max is not None:
        max = torch.as_tensor(max).expand_as(tensor)
        expected = torch.min(max, expected)

    assert_equal(actual, expected)


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (2, 3, 4)], ids=str)
def test_cat2_scalar(shape):
    tensor = torch.randn(shape)
    for dim in range(-len(shape), 0):
        expected_shape = list(shape)
        expected_shape[dim] += 1
        expected_shape = torch.Size(expected_shape)
        assert cat2(tensor, 0, dim=dim).shape == expected_shape
        assert cat2(0, tensor, dim=dim).shape == expected_shape
        assert (cat2(tensor, 0, dim=dim).unbind(dim)[-1] == 0).all()
        assert (cat2(0, tensor, dim=dim).unbind(dim)[0] == 0).all()


@pytest.mark.parametrize("shape", [(), (8,), (5, 3)])
@pytest.mark.parametrize("num_quant_bins", [2, 4, 8])
def test_quantize_enumerate_pathwise(shape, num_quant_bins):
    min = 0
    max = 20
    x_real = torch.rand(shape).mul(max).requires_grad_()

    m0 = torch.randn(shape + (1,))
    m1 = torch.randn(shape + (1,)).add(1).mul(max)
    m2 = torch.rand(shape + (1,)).sub(0.5).div(max - min)

    def test_fn(x, logits):
        y = m0 + (x - m1).pow(2) * m2
        probs = logits.exp()
        return (probs * y).sum()

    x, logits = quantize_enumerate(x_real, min, max, num_quant_bins)
    expected = test_fn(x, logits)
    expected_grad = grad(expected, [x_real], retain_graph=True)

    x, logits = quantize_enumerate(x_real, min, max, num_quant_bins,
                                   pathwise_radius=1)
    actual = test_fn(x, logits)
    actual_grad = grad(actual, [x_real], retain_graph=True)

    assert_equal(actual, expected)
    assert_close(actual_grad, expected_grad, atol=0.05)
