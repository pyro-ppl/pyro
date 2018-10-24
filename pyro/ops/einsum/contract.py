from __future__ import absolute_import, division, print_function

import importlib
from collections import Counter

import torch
from opt_einsum.sharing import einsum_cache_wrap

from pyro.ops.einsum.paths import optimize


@einsum_cache_wrap
def _einsum(equation, *operands, **kwargs):
    backend = kwargs.pop('backend', 'numpy')
    if backend == 'torch':
        # provide np.einsum interface for torch.einsum.
        return torch.einsum(equation, operands)
    einsum = getattr(importlib.import_module(backend), 'einsum')
    return einsum(equation, *operands, **kwargs)


class ContractExpression(object):
    def __init__(self, equation, *shapes):
        self.equation = equation
        self.shapes = shapes

        inputs, output = equation.split('->')
        inputs = inputs.split(',')
        sizes = {dim: size for dims, shape in zip(inputs, shapes)
                 for dim, size in zip(dims, shape)}
        self.path = optimize(inputs, output, sizes)

    def __call__(self, *operands, **kwargs):
        out = kwargs.pop('out', None)

        ref_counts = Counter(self.equation)
        inputs, output = self.equation.split('->')
        inputs = inputs.split(',')
        remaining = list(zip(inputs, operands))
        for op in self.path:
            op_inputs = []
            op_tensors = []
            for i in sorted(op, reverse=True):
                dims, tensor = remaining.pop(i)
                op_tensors.append(tensor)
                op_inputs.append(dims)
                ref_counts.subtract(dims)
            if remaining:
                op_output = ''.join(sorted(d for d in set(''.join(op_inputs)) if ref_counts[d]))
            else:
                op_output = output
            ref_counts.update(op_output)
            op_equation = ','.join(op_inputs) + '->' + op_output
            tensor = _einsum(op_equation, *op_tensors, **kwargs)
            remaining.append((op_output, tensor))
        assert len(remaining) == 1

        result = remaining[0][1]
        if out is not None:
            out.copy_(result)
        return result
