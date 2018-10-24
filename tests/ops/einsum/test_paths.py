from __future__ import absolute_import, division, print_function

import logging
import re
import timeit

import opt_einsum
import pytest
import torch
from six import text_type

from pyro.ops.einsum.paths import linear_to_ssa, optimize, ssa_to_linear
from tests.common import assert_equal
from tests.ops.einsum.data import EQUATIONS, LARGE, SIZES, make_shapes


def _test_path(equation, shapes):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    dim_sizes = {dim: size for dims, shape in zip(inputs, shapes)
                 for dim, size in zip(dims, shape)}
    operands = [torch.randn(shape) for shape in shapes]
    logging.debug(equation)

    # Compute path using Pyro.
    pyro_time = -timeit.default_timer()
    pyro_path = optimize(inputs, output, dim_sizes)
    pyro_time += timeit.default_timer()

    assert sum(map(len, pyro_path)) - len(pyro_path) + 1 == len(inputs)
    path = ssa_to_linear(linear_to_ssa(pyro_path))
    assert path == pyro_path

    # Compute path using opt_einsum's greedy method.
    opt_time = -timeit.default_timer()
    opt_path, opt_info = opt_einsum.contract_path(equation, *operands, path='greedy')
    opt_time += timeit.default_timer()

    # Check path quality.
    _, pyro_info = opt_einsum.contract_path(equation, *operands, path=pyro_path)
    # Wrap in text_type to avoid breaking after https://github.com/dgasmith/opt_einsum/pull/61
    pyro_info = '\n'.join(text_type(pyro_info).splitlines()[1:7])
    opt_info = '\n'.join(text_type(opt_info).splitlines()[1:7])
    logging.debug(u'Pyro path took {}s:\n{}'.format(pyro_time, pyro_info))
    logging.debug(u'opt_einsum took {}s:\n{}'.format(opt_time, opt_info))
    pyro_flops = float(re.search('Optimized FLOP count:(.*)', pyro_info).group(1))
    opt_flops = float(re.search('Optimized FLOP count:(.*)', opt_info).group(1))
    assert pyro_flops <= opt_flops * 1.4

    # Check path correctness.
    try:
        expected = opt_einsum.contract(equation, *operands, backend='torch', optimize=opt_path)
    except RuntimeError:
        return  # ignore torch not implemented errors
    actual = opt_einsum.contract(equation, *operands, backend='torch', optimize=pyro_path)
    assert_equal(expected, actual)


@pytest.mark.parametrize('sizes', SIZES)
@pytest.mark.parametrize('equation', EQUATIONS)
def test_contract(equation, sizes):
    shapes = make_shapes(equation, sizes)
    _test_path(equation, shapes)


def test_contract_large():
    _test_path(LARGE['equation'], LARGE['shapes'])


if __name__ == '__main__':
    test_contract_large()
