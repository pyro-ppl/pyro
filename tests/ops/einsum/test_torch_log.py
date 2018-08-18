from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.ops.einsum import contract
from tests.common import assert_equal


@pytest.mark.parametrize('equation', [
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
    operands = [torch.randn(*(sizes[dim] for dim in dims))
                for dims in inputs]

    expected = contract(equation, *(d.exp() for d in operands), backend='torch').log()
    actual = contract(equation, *operands, backend='pyro.ops.einsum.torch_log')
    assert_equal(actual, expected)
