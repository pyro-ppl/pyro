# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

from pyro.infer.util import torch_exp
from pyro.ops.einsum import contract
from tests.common import assert_equal


@pytest.mark.parametrize('min_size', [1, 2])
@pytest.mark.parametrize('equation', [
    ',ab->ab',
    'ab,,bc->a',
    'ab,,bc->b',
    'ab,,bc->c',
    'ab,,bc->ac',
    'ab,,b,bc->ac',
    'a,ab->ab',
    'ab,b,bc->a',
    'ab,b,bc->b',
    'ab,b,bc->c',
    'ab,b,bc->ac',
    'ab,bc->ac',
    'ab,bc,cd->',
    'ab,bc,cd->a',
    'ab,bc,cd->b',
    'ab,bc,cd->c',
    'ab,bc,cd->d',
    'ab,bc,cd->ac',
    'ab,bc,cd->ad',
    'ab,bc,cd->bc',
    'a,a,ab,b,b,b,b->a',
])
@pytest.mark.parametrize('infinite', [False, True], ids=['finite', 'infinite'])
def test_einsum(equation, min_size, infinite):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    symbols = sorted(set(equation) - set(',->'))
    sizes = dict(zip(symbols, itertools.count(min_size)))
    shapes = [torch.Size(tuple(sizes[dim] for dim in dims))
              for dims in inputs]
    operands = [torch.full(shape, -float('inf')) if infinite else torch.randn(shape)
                for shape in shapes]

    expected = contract(equation, *(torch_exp(x) for x in operands), backend='torch').log()
    actual = contract(equation, *operands, backend='pyro.ops.einsum.torch_log')
    assert_equal(actual, expected)
