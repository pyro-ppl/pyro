from __future__ import absolute_import, division, print_function

import logging
import timeit

import opt_einsum
import pytest
import torch

from pyro.ops.einsum import contract
from pyro.ops.einsum.contract import ContractExpression
from pyro.ops.einsum.paths import optimize
from tests.common import assert_equal
from tests.ops.einsum.data import EQUATIONS, LARGE, SIZES, make_shapes

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


def test_large():
    equation = LARGE['equation']
    shapes = LARGE['shapes']
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    dim_sizes = {dim: size for dims, shape in zip(inputs, shapes)
                 for dim, size in zip(dims, shape)}
    operands = [torch.randn(shape) for shape in shapes]

    # test pyro contraction
    expr = ContractExpression(equation, *shapes)
    pyro_time = -timeit.default_timer()
    actual = expr(*operands, backend=backend)
    pyro_time += timeit.default_timer()

    # test opt_einsum contraction
    path = optimize(inputs, output, dim_sizes)
    expr = opt_einsum.contract_expression(equation, *shapes, optimize=path)
    opt_time = -timeit.default_timer()
    expected = expr(*operands, backend=backend)
    opt_time += timeit.default_timer()

    logging.debug(u'Pyro contract took {}s'.format(pyro_time))
    logging.debug(u'opt_einsum contract took {}s'.format(opt_time))
    assert_equal(expected, actual)
