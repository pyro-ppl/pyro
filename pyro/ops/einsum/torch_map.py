from __future__ import absolute_import, division, print_function

import operator

from six.moves import reduce

from pyro.ops import packed
from pyro.ops.einsum.adjoint import Backward, einsum_backward_sample, transpose, unflatten
from pyro.ops.einsum.util import Tensordot

assert transpose  # pacify flake8


class _EinsumBackward(Backward):
    def __init__(self, operands, argmax):
        self.operands = operands
        self.argmax = argmax

    def process(self, message):
        sample1 = self.argmax
        sample2 = message
        return einsum_backward_sample(self.operands, sample1, sample2)


def einsum(equation, *operands):
    """
    Forward-max-sum backward-argmax implementation of einsum.
    """
    equation = packed.rename_equation(equation, *operands)
    inputs, output = equation.split('->')
    any_requires_backward = any(hasattr(x, '_pyro_backward') for x in operands)

    contract_dims = ''.join(sorted(set().union(*(x._pyro_dims for x in operands)) - set(output)))
    dims = output + contract_dims
    result = reduce(operator.add, packed.broadcast_all(*operands, dims=dims))
    argmax = None  # work around lack of pytorch support for zero-sized tensors
    if contract_dims:
        output_shape = result.shape[:len(output)]
        contract_shape = result.shape[len(output):]
        result, argmax = result.reshape(output_shape + (-1,)).max(-1)
        if any_requires_backward:
            argmax = unflatten(argmax, output, contract_dims, contract_shape)
    result._pyro_dims = output
    assert result.dim() == len(result._pyro_dims)

    if any_requires_backward:
        result._pyro_backward = _EinsumBackward(operands, argmax)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
