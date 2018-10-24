from __future__ import absolute_import, division, print_function

import opt_einsum
import pytest
import torch

from pyro.ops.einsum import contract
from tests.common import assert_equal
from tests.ops.einsum.data import EQUATIONS, SIZES, make_shapes

backend = 'pyro.ops.einsum.torch_log'


@pytest.mark.parametrize('sizes', SIZES)
@pytest.mark.parametrize('equation', EQUATIONS)
def test_contract(equation, sizes):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    if any(len(set(dims)) < len(dims) for dims in inputs):
        pytest.xfail(reason='torch.einsum does not support repeated indices in a single tensor')

    shapes = make_shapes(equation, sizes)
    operands = [torch.randn(shape) for shape in shapes]
    expected = opt_einsum.contract(equation, *operands, backend=backend)
    actual = contract(equation, *operands, backend=backend)
    assert_equal(expected, actual)
