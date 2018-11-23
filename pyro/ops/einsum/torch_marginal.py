from __future__ import absolute_import, division, print_function

import torch

import pyro.ops.einsum.torch_log
from pyro.ops.einsum.adjoint import Backward, transpose
from pyro.ops.einsum.util import Tensordot

assert transpose  # pacify flake8


class _EinsumBackward(Backward):
    def __init__(self, equation, operands):
        self.equation = equation
        self.operands = operands

    def process(self, message):
        operands = list(self.operands)
        inputs, output = self.equation.split("->")
        inputs = inputs.split(",")
        if message is None:
            sizes = {dim: size
                     for input_, operand in zip(inputs, operands)
                     for dim, size in zip(input_, operand.shape)}
            message_shape = torch.Size(sizes[dim] for dim in output)
            message = self.operands[0].new_zeros(message_shape)
        assert message.dim() == len(output)
        for i in range(len(inputs)):
            if not hasattr(operands[i], "_pyro_backward"):
                continue
            # swap output <-> inputs[i]
            inputs[i], output = output, inputs[i]
            operands[i], message = message, operands[i]

            # pass a message to inputs[i]
            equation = ",".join(inputs) + "->" + output
            message_i = pyro.ops.einsum.torch_log.einsum(equation, *operands)
            message_i = message_i.expand_as(message)
            yield message._pyro_backward, message_i

            # swap back
            inputs[i], output = output, inputs[i]
            operands[i], message = message, operands[i]


def einsum(equation, *operands):
    """
    Forward-log-sum-product-exp backward-marginal implementation of einsum.
    """
    result = pyro.ops.einsum.torch_log.einsum(equation, *operands)
    if any(hasattr(x, '_pyro_backward') for x in operands):
        result._pyro_backward = _EinsumBackward(equation, operands)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
