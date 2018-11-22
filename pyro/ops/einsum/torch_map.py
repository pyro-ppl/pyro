from __future__ import absolute_import, division, print_function

import operator

from six.moves import reduce

from pyro.ops.einsum.adjoint import einsum_backward_scatter, requires_backward, transpose, unflatten  # noqa F403
from pyro.ops.einsum.util import Tensordot, einbroadcast


class _EinsumBackward(object):
    def __init__(self, inputs, operands, argmax):
        self.inputs = inputs
        self.operands = operands
        self.argmax = argmax

    def __call__(self, sample2=None):
        sample1 = self.argmax
        einsum_backward_scatter(self.inputs, self.operands, sample1, sample2)


def einsum(equation, *operands):
    """
    Max-sum implementation of einsum (aka tropical einsum).
    """
    inputs, output = equation.split("->")
    inputs = inputs.split(",")
    result_requires_backward = any(requires_backward(x) for x in operands)

    contract_dims = "".join(sorted(set().union(*inputs) - set(output)))
    dims = output + contract_dims
    result = reduce(operator.add, einbroadcast(inputs, dims, operands))
    argmax = None  # work around lack of pytorch support for zero-sized tensors
    if contract_dims:
        output_shape = result.shape[:len(output)]
        contract_shape = result.shape[len(output):]
        result, argmax = result.reshape(output_shape + (-1,)).max(-1)
        if result_requires_backward:
            argmax = unflatten(argmax, output, contract_dims, contract_shape)

    if result_requires_backward:
        result._pyro_backward = _EinsumBackward(equation, operands, argmax)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
