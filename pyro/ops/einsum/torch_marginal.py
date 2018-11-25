from __future__ import absolute_import, division, print_function

import pyro.ops.einsum.torch_log
from pyro.ops.einsum.adjoint import Backward, transpose
from pyro.ops.einsum.util import Tensordot

assert transpose  # pacify flake8


class _EinsumBackward(Backward):
    def __init__(self, equation, operands):
        self.equation = equation
        self.operands = operands

    def process(self, message):
        # Create extended lists of inputs and operands.
        operands = list(self.operands)
        inputs, output = self.equation.split('->')
        inputs = inputs.split(',')
        if message is not None:
            assert message.dim() == len(output)
            inputs.append(output)
            operands.append(message)

        # Aggregate all messages and pass backward.
        for i, operand in enumerate(self.operands):
            if not hasattr(operand, "_pyro_backward"):
                continue
            output_i = inputs[i]
            inputs_i = list(inputs)
            operands_i = list(operands)
            if not operand._pyro_backward.is_leaf:
                del inputs_i[i]
                del operands_i[i]
            equation = ','.join(inputs_i) + '->' + output_i
            message_i = pyro.ops.einsum.torch_log.einsum(equation, *operands_i)
            yield operand._pyro_backward, message_i


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
