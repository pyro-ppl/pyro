from __future__ import absolute_import, division, print_function

import operator

from six.moves import reduce

import pyro.distributions as dist
import pyro.ops.einsum.torch_log
from pyro.ops import packed
from pyro.ops.einsum.adjoint import einsum_backward_scatter, requires_backward, transpose, unflatten  # noqa F403
from pyro.ops.einsum.util import Tensordot, einbroadcast


class _EinsumBackward(object):
    def __init__(self, equation, operands):
        self.equation = equation
        self.operands = operands

    def __call__(self, sample2=None):
        operands = self.operands
        inputs, output = self.equation.split("->")
        inputs = inputs.split(",")
        contract_dims = "".join(sorted(set.union(*inputs) - set(output)))
        dims = output + contract_dims

        # Slice down operands before combining terms.
        if sample2 is not None:
            for i, (input_, x) in enumerate(zip(inputs, operands)):
                for dim, index in zip(sample2._pyro_sample_dims, sample2):
                    if dim in input_:
                        x = packed.gather(x, index, dim)
                        input_ = input_.replace(dim, '')
                    inputs[i] = input_
                    operands[i] = x

        # Combine terms.
        logits = reduce(operator.add, einbroadcast(inputs, dims, operands))
        logits._pyro_dims = dims
        assert logits.dim() == len(logits._pyro_dims)

        # Sample.
        sample1 = None  # work around lack of pytorch support for zero-sized tensors
        if contract_dims:
            output_shape = logits.shape[:len(output)]
            contract_shape = logits.shape[len(output):]
            flat_logits = logits.reshape(output_shape + (-1))
            flat_sample = dist.Categorical(logits=flat_logits).sample()
            sample1 = unflatten(flat_sample, output, contract_dims, contract_shape)

        # Cut down samples to pass on to subsequent steps.
        einsum_backward_scatter(self.operands, sample1, sample2)


def einsum(equation, *operands):
    result = pyro.ops.einsum.torch_log.einsum(equation, *operands)
    if any(requires_backward(x) for x in operands):
        result._pyro_backward = _EinsumBackward(equation, operands)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
