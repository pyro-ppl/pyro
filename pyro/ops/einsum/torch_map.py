from __future__ import absolute_import, division, print_function

import operator

from six.moves import reduce

from pyro.ops.einsum.adjoint import requires_backward, transpose  # noqa F403
from pyro.ops.einsum.util import Tensordot, einbroadcast


class _EinsumBackward(object):
    def __init__(self, equation, operands, argmax):
        self.equation = equation
        self.operands = operands
        self.argmax = argmax

    def __call__(self, sample):
        raise NotImplementedError("TODO")


def einsum(equation, *operands):
    inputs, output = equation.split("->")
    inputs = inputs.split(",")

    contract_dims = "".join(sorted(set.union(*inputs) - set(output)))
    dims = output + contract_dims
    result = reduce(operator.add, einbroadcast(inputs, dims, operands))
    argmax = None
    while dims != output:
        result, argmax_dim = result.max(-1)
        dims = dims[:-1]
        raise NotImplementedError("TODO")

    if any(requires_backward(x) for x in operands):
        result._pyro_backward = _EinsumBackward(equation, operands, argmax)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
