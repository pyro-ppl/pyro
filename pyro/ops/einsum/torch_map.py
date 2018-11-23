from __future__ import absolute_import, division, print_function

import operator

from six.moves import reduce

from pyro.ops.einsum.adjoint import Backward, einsum_backward_sample, transpose, unflatten
from pyro.ops.einsum.util import Tensordot, einbroadcast

assert transpose  # pacify flake8


class _EinsumBackward(Backward):
    def __init__(self, inputs, operands, argmax):
        self.inputs = inputs
        self.operands = operands
        self.argmax = argmax

    def process(self, message):
        sample1 = self.argmax
        sample2 = message
        return einsum_backward_sample(self.inputs, self.operands, sample1, sample2)


def einsum(equation, *operands):
    """
    Forward-max-sum backward-argmax implementation of einsum.
    """
    inputs, output = equation.split("->")
    inputs = inputs.split(",")
    any_requires_backward = any(hasattr(x, '_pyro_backward') for x in operands)

    contract_dims = "".join(sorted(set().union(*inputs) - set(output)))
    dims = output + contract_dims
    result = reduce(operator.add, einbroadcast(inputs, dims, operands))
    argmax = None  # work around lack of pytorch support for zero-sized tensors
    if contract_dims:
        output_shape = result.shape[:len(output)]
        contract_shape = result.shape[len(output):]
        result, argmax = result.reshape(output_shape + (-1,)).max(-1)
        if any_requires_backward:
            argmax = unflatten(argmax, output, contract_dims, contract_shape)

    if any_requires_backward:
        result._pyro_backward = _EinsumBackward(equation, operands, argmax)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
