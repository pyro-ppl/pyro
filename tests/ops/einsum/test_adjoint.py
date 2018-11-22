from __future__ import absolute_import, division, print_function

import itertools

import pytest
import torch

from pyro.ops.einsum import contract
from pyro.ops.einsum.adjoint import require_backward

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


@pytest.mark.parametrize('min_size', [1, 2])
@pytest.mark.parametrize('equation', EQUATIONS)
@pytest.mark.parametrize('backend', ['map', 'sample'])
def test_shape(backend, equation, min_size):
    backend = 'pyro.ops.einsum.torch_{}'.format(backend)
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    symbols = sorted(set(equation) - set(',->'))
    sizes = dict(zip(symbols, itertools.count(min_size)))
    output_shape = torch.Size(sizes[dim] for dim in output)
    input_shapes = [torch.Size(sizes[dim] for dim in dims)
                    for dims in inputs]
    operands = [torch.randn(shape) for shape in input_shapes]

    # run forward-backward algorithm
    for x in operands:
        require_backward(x)
    result = contract(equation, *operands, backend=backend)
    result._pyro_backward()

    for input_, x in zip(inputs, operands):
        backward_result = x._pyro_backward_result
        contract_dims = set(input_) - set(output)
        if contract_dims:
            assert backward_result.shape == (len(contract_dims),) + output_shape
        else:
            assert backward_result is None
