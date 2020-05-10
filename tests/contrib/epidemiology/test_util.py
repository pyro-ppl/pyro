# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.contrib.epidemiology.util import cat2, clamp
from tests.common import assert_equal


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
