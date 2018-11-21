from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
import pyro.ops.einsum.torch_log
from pyro.ops import packed
from pyro.ops.einsum.adjoint import requires_backward, transpose  # noqa F403
from pyro.ops.einsum.util import Tensordot


class _EinsumBackward(object):
    def __init__(self, equation, operands):
        self.equation = equation
        self.operands = operands

    def __call__(self, sample):
        logits = self.logits
        if sample is None:
            for i in range(sample.dim(-1)):
                dim = sample._pyro_sample_dims[i]
                index = sample[i]
                logits = packed.gather(logits, index, dim)

        flat_logits = logits.reshape('todo (-1,) + output_shape')
        flat_sample = dist.categorical(logits=flat_logits)
        if flat_logits.dim() == self.logits.dim():
            sample = flat_sample.unsqueeze(0)
        else:
            raise NotImplementedError('todo modular arithmetic; torch.stack()')

        for x in self.operands:
            if requires_backward(x):
                continue
            needed_dims = set(x._pyro_dims) & set(sample._pyro_sample_dims)
            needed_idx = [sample._pyro_sample_dims.index(dim) for dim in x._pyro_dims]
            result = sample[needed_idx]
            if hasattr(x, '_pyro_backward'):
                x._pyro_backward(result)
            else:  # a leaf variable
                x._pyro_backward_result = result


def einsum(equation, *operands):
    result = pyro.ops.einsum.torch_log.einsum(equation, *operands)
    if any(requires_backward(x) for x in operands):
        result._pyro_backward = _EinsumBackward(equation, operands)
    return result


tensordot = Tensordot(einsum)

__all__ = ["transpose", "einsum", "tensordot"]
