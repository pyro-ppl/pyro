from __future__ import absolute_import, division, print_function

import itertools

import pytest
import torch

from pyro.ops.einsum import contract
from pyro.ops.einsum.adjoint import require_backward
from tests.common import assert_equal

EQUATIONS = [
    'a->',
    ',a->',
    'a,a->',
    'a,b->',
    'a,ab,b->',
    'a,ab,bc,cd->',
    'ai->i',
    'i,ai->i',
    'ai,ai->i',
    'ai,bi->i',
    'ai,abi,bi->i',
    'ai,abi,bci,cdi->i',
    'iaj->ij',
    'ij,iaj->ij',
    'iaj,iaj->ij',
    'iaj,ibj->ij',
    'iaj,iabj,ibj->ij',
    'iaj,iabj,ibcj,icdj->ij',
]


@pytest.mark.parametrize('equation', EQUATIONS)
@pytest.mark.parametrize('backend', ['map', 'sample', 'marginal'])
def test_shape(backend, equation):
    backend = 'pyro.ops.einsum.torch_{}'.format(backend)
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    symbols = sorted(set(equation) - set(',->'))
    sizes = dict(zip(symbols, itertools.count(2)))
    output_shape = torch.Size(sizes[dim] for dim in output)
    input_shapes = [torch.Size(sizes[dim] for dim in dims)
                    for dims in inputs]
    operands = [torch.randn(shape) for shape in input_shapes]

    # check forward pass
    for x in operands:
        require_backward(x)
    expected = contract(equation, *operands, backend='pyro.ops.einsum.torch_log')
    actual = contract(equation, *operands, backend=backend)
    if backend.endswith('map'):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
    else:
        assert_equal(actual, expected)

    # check backward pass
    actual._pyro_backward()
    for input_, x in zip(inputs, operands):
        backward_result = x._pyro_backward_result
        if backend.endswith('marginal'):
            assert backward_result.shape == x.shape
        else:
            contract_dims = set(input_) - set(output)
            if contract_dims:
                assert backward_result.shape == (len(contract_dims),) + output_shape
            else:
                assert backward_result is None


@pytest.mark.parametrize('equation', EQUATIONS)
def test_marginal(equation):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    operands = [torch.randn(torch.Size((2,) * len(dims)))
                for dims in inputs]

    # check forward pass
    for x in operands:
        require_backward(x)
    actual = contract(equation, *operands, backend='pyro.ops.einsum.torch_marginal')
    expected = contract(equation, *operands,
                        backend='pyro.ops.einsum.torch_log')
    assert_equal(expected, actual, msg='\n'.join([
        'Expected:', str(expected.detach().cpu().numpy()),
        'Actual:', str(actual.detach().cpu().numpy()),
    ]))

    # check backward pass
    actual._pyro_backward()
    for input_, operand in zip(inputs, operands):
        marginal_equation = ','.join(inputs) + '->' + input_
        expected = contract(marginal_equation, *operands,
                            backend='pyro.ops.einsum.torch_log')
        actual = operand._pyro_backward_result
        assert_equal(expected, actual, msg='\n'.join([
            'Expected:', str(expected.detach().cpu().numpy()),
            'Actual:', str(actual.detach().cpu().numpy()),
        ]))
