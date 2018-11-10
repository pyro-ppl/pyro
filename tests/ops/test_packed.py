from __future__ import absolute_import, division, print_function

import random

import pytest
import torch
from torch.distributions.utils import broadcast_all

from pyro.ops import packed
from pyro.ops.sumproduct import logsumproductexp, sumproduct
from tests.common import assert_equal

EXAMPLE_SHAPES = [
    [],
    [()],
    [(), ()],
    [(2,), (3, 1)],
    [(2,), (3, 1), (3, 2)],
]


def make_inputs(shapes, num_numbers=0):
    inputs = [torch.randn(shape) for shape in shapes]
    num_symbols = max(map(len, shapes)) if shapes else 0
    for _ in range(num_numbers):
        inputs.append(random.random())
    dim_to_symbol = {}
    symbol_to_dim = {}
    for dim, symbol in zip(range(-num_symbols, 0), 'abcdefghijklmnopqrstuvwxyz'):
        dim_to_symbol[dim] = symbol
        symbol_to_dim[symbol] = dim
    return inputs, dim_to_symbol, symbol_to_dim


@pytest.mark.parametrize('shapes', EXAMPLE_SHAPES)
def test_broadcast_all(shapes):
    inputs, dim_to_symbol, symbol_to_dim = make_inputs(shapes)
    packed_inputs = [packed.pack(x, dim_to_symbol) for x in inputs]
    packed_outputs = packed.broadcast_all(*packed_inputs)
    actual = tuple(packed.unpack(x, symbol_to_dim) for x in packed_outputs)
    expected = broadcast_all(*inputs)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert_equal(a, e)


@pytest.mark.parametrize('shapes', EXAMPLE_SHAPES)
@pytest.mark.parametrize('num_numbers', [0, 1, 2])
def test_sumproduct(shapes, num_numbers):
    inputs, dim_to_symbol, symbol_to_dim = make_inputs(shapes, num_numbers)
    packed_inputs = [packed.pack(x, dim_to_symbol) for x in inputs]
    packed_output = packed.sumproduct(packed_inputs)
    actual = packed.unpack(packed_output, symbol_to_dim)
    expected = sumproduct(inputs)
    assert_equal(actual, expected)


@pytest.mark.parametrize('shapes', EXAMPLE_SHAPES)
@pytest.mark.parametrize('num_numbers', [0, 1, 2])
def test_logsumproductexp(shapes, num_numbers):
    inputs, dim_to_symbol, symbol_to_dim = make_inputs(shapes, num_numbers)
    packed_inputs = [packed.pack(x, dim_to_symbol) for x in inputs]
    packed_output = packed.logsumproductexp(packed_inputs)
    actual = packed.unpack(packed_output, symbol_to_dim)
    expected = logsumproductexp(inputs)
    assert_equal(actual, expected)
