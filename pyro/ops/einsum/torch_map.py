from __future__ import absolute_import, division, print_function

import operator

import torch
from six.moves import reduce

from pyro.ops import packed
from pyro.ops.einsum.adjoint import requires_backward, transpose  # noqa F403
from pyro.ops.einsum.util import Tensordot, einbroadcast

SAMPLE_DIM = " "


class _EinsumBackward(object):
    def __init__(self, equation, operands, argmax):
        self.equation = equation
        self.operands = operands
        self.argmax = argmax

    def __call__(self, sample2=None):
        sample1 = self.argmax
        if sample1 is None:
            sample = sample2
        elif sample2 is None:
            sample = sample1
        else:
            sample = sample1
            for dim, index in zip(sample2._pyro_sample_dims, sample2):
                sample = packed.gather(sample, index, dim)
            parts = packed.broadcast_all(sample, sample2)
            sample = torch.cat(parts)
            sample._pyro_dims = parts[0]._pyro_dims
            sample._pyro_sample_dims = sample1._pyro_sample_dims + sample2._pyro_sample_dims
            assert sample.dim() == len(sample._pyro_dims)
            assert sample.size(0) == len(sample._pyro_sample_dims)

        for x in self.operands:
            if not requires_backward(x):
                continue
            # TODO cut down to required dims
            if hasattr(x, "_pyro_backward"):
                x._pyro_backward(sample)
            else:  # a leaf variable
                x._pyro_backward_result = sample


def einsum(equation, *operands):
    inputs, output = equation.split("->")
    inputs = inputs.split(",")
    result_requires_backward = any(requires_backward(x) for x in operands)

    contract_dims = "".join(sorted(set.union(*inputs) - set(output)))
    dims = output + contract_dims
    result = reduce(operator.add, einbroadcast(inputs, dims, operands))
    argmax = None  # work around lack of pytorch support for zero-sized tensors
    if contract_dims:
        output_shape = result.shape[:len(output)]
        contract_shape = result.shape[len(output):]
        result, argmax = result.reshape(output_shape + (-1,)).max(-1)
        if result_requires_backward:
            argmax = argmax.unsqueeze(0)
            if len(contract_dims) > 1:
                slices = [None] * len(contract_dims)
                for i, size in reversed(enumerate(contract_shape)):
                    slices[i] = argmax % size
                    argmax /= size
                argmax = torch.cat(slices)
            argmax._pyro_dims = SAMPLE_DIM + output
            argmax._pyro_sample_dims = contract_dims
            assert argmax.dims() == len(argmax._pyro_dims)
            assert argmax.size(0) == len(argmax._pyro_sample_dims)

    if result_requires_backward:
        result._pyro_backward = _EinsumBackward(equation, operands, argmax)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
