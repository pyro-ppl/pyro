# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import operator

from functools import reduce

import pyro.distributions as dist
import pyro.ops.einsum.torch_log
from pyro.ops import packed
from pyro.ops.einsum.adjoint import Backward, einsum_backward_sample, transpose, unflatten
from pyro.ops.einsum.util import Tensordot
from pyro.util import jit_iter


class _EinsumBackward(Backward):
    def __init__(self, output, operands):
        self.output = output
        self.operands = operands

    def process(self, message):
        output = self.output
        operands = list(self.operands)
        contract_dims = ''.join(sorted(set().union(*(x._pyro_dims for x in operands)) - set(output)))
        batch_dims = output

        # Slice down operands before combining terms.
        sample2 = message
        if sample2 is not None:
            for dim, index in zip(sample2._pyro_sample_dims, jit_iter(sample2)):
                batch_dims = batch_dims.replace(dim, '')
                for i, x in enumerate(operands):
                    if dim in x._pyro_dims:
                        index._pyro_dims = sample2._pyro_dims[1:]
                        x = packed.gather(x, index, dim)
                    operands[i] = x

        # Combine terms.
        dims = batch_dims + contract_dims
        logits = reduce(operator.add, packed.broadcast_all(*operands, dims=dims))

        # Sample.
        sample1 = None  # work around lack of pytorch support for zero-sized tensors
        if contract_dims:
            output_shape = logits.shape[:len(batch_dims)]
            contract_shape = logits.shape[len(batch_dims):]
            flat_logits = logits.reshape(output_shape + (-1,))
            flat_sample = dist.Categorical(logits=flat_logits).sample()
            sample1 = unflatten(flat_sample, batch_dims, contract_dims, contract_shape)

        # Cut down samples to pass on to subsequent steps.
        return einsum_backward_sample(self.operands, sample1, sample2)


def einsum(equation, *operands):
    """
    Forward-log-sum-product-exp backward-sample-exp implementation of einsum.
    This assumes all operands have a ``._pyro_dims`` attribute set.
    """
    equation = packed.rename_equation(equation, *operands)
    inputs, output = equation.split('->')
    result = pyro.ops.einsum.torch_log.einsum(equation, *operands)
    result._pyro_dims = output
    assert result.dim() == len(result._pyro_dims)

    if any(hasattr(x, '_pyro_backward') for x in operands):
        result._pyro_backward = _EinsumBackward(output, operands)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
