from __future__ import absolute_import, division, print_function

import operator

import pytest
import torch
from six.moves import reduce

from pyro.ops.sumproduct import sumproduct, zip_align_right
from tests.common import assert_equal


@pytest.mark.parametrize('xs,ys,expected', [
    (['a', 'b'], [6, 5], [('a', 6), ('b', 5)]),
    (['b'], [6, 5], [('b', 5)]),
    (['a', 'b'], [5], [('b', 5)]),
])
def test_zip_align_right(xs, ys, expected):
    actual = list(zip_align_right(xs, ys))
    assert actual == expected


@pytest.mark.parametrize('factor_shapes,shape', [
    ([(), (100000, 1, 2, 1), (2, 100000, 1, 1, 1)], (100000, 1, 2, 1)),
    ([None, (100000, 1, 2, 1), (2, 100000, 1, 1, 1)], (100000, 1, 2, 1)),
])
@pytest.mark.parametrize('optimize', [False, True])
def test_sumproduct(factor_shapes, shape, optimize):
    factors = [0.2 if s is None else torch.randn(s).exp()
               for s in factor_shapes]
    actual = sumproduct(factors, shape, optimize=optimize)

    expected = reduce(operator.mul, factors)
    while expected.dim() > len(shape):
        expected = expected.sum(0)
    while expected.dim() < len(shape):
        expected = expected.unsqueeze(0)
    for i, (e, s) in enumerate(zip(expected.shape, shape)):
        if e > s:
            expected = expected.sum(i, keepdim=True)
    assert expected.shape == shape

    assert_equal(actual, expected)
