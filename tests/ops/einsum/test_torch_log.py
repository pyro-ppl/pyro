from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.infer.util import torch_exp
from pyro.ops.einsum import contract
from pyro.ops.sumproduct import logsumproductexp, sumproduct
from tests.common import assert_equal


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
])
def test_einsum(equation):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    symbols = sorted(set(equation) - set(',->'))
    sizes = dict(zip(symbols, range(2, 2 + len(symbols))))
    shapes = [torch.Size(tuple(sizes[dim] for dim in dims))
              for dims in inputs]
    operands = [torch.randn(shape) for shape in shapes]

    expected = contract(equation, *(torch_exp(x) for x in operands), backend='torch').log()
    actual = contract(equation, *operands, backend='pyro.ops.einsum.torch_log')
    assert_equal(actual, expected)


@pytest.mark.parametrize('shapes', [
    ((), (1,), (1, 2)),
    ((), (2,)),
    ((1,), (1, 2)),
    ((1,), (2,)),
    ((2, 1), (1, 3)),
    ((2, 1), (2, 3)),
    ((2, 3), (2, 3)),
    ((3,), (2, 3)),
    ((4, 1, 1), None, (), (2, 3)),
    ((4, 1, 1), None, None, (2, 3)),
    (None, (1,), (1, 2)),
    (None, (2,)),
])
def test_logsumproductexp(shapes):
    factors = [float(torch.randn(torch.Size())) if shape is None else torch.randn(shape)
               for shape in shapes]
    expected = sumproduct([torch_exp(x) for x in factors]).log()
    actual = logsumproductexp(factors)
    assert_equal(actual, expected)
