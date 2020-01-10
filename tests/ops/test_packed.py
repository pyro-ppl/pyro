# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import random

import pytest
import torch
from torch.distributions.utils import broadcast_all

from pyro.ops import packed
from tests.common import assert_equal

EXAMPLE_DIMS = [
    ''.join(dims)
    for num_dims in range(5)
    for dims in itertools.permutations('abcd'[:num_dims])
]


@pytest.mark.parametrize('dims', EXAMPLE_DIMS)
def test_unpack_pack(dims):
    dim_to_symbol = {}
    symbol_to_dim = {}
    for symbol, dim in zip('abcd', range(-1, -5, -1)):
        dim_to_symbol[dim] = symbol
        symbol_to_dim[symbol] = dim
    shape = tuple(range(2, 2 + len(dims)))
    x = torch.randn(shape)

    pack_x = packed.pack(x, dim_to_symbol)
    unpack_pack_x = packed.unpack(pack_x, symbol_to_dim)
    assert_equal(unpack_pack_x, x)

    sort_dims = ''.join(sorted(dims))
    if sort_dims != pack_x._pyro_dims:
        sort_pack_x = pack_x.permute(*(pack_x._pyro_dims.index(d) for d in sort_dims))
        sort_pack_x._pyro_dims = sort_dims
        unpack_sort_pack_x = packed.unpack(sort_pack_x, symbol_to_dim)
        assert_equal(unpack_sort_pack_x, x)


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
    expected = broadcast_all(*inputs) if inputs else []
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert_equal(a, e)
