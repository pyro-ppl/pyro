from __future__ import absolute_import, division, print_function

import importlib
from collections import Counter

import torch

from pyro.ops.einsum.paths import optimize


def torch_einsum(equation, *operands):
    return torch.einsum(equation, operands)


def ubersum_step(equation, *operands, **kwargs):
    einsum = kwargs.get('einsum')
    batch_dims = kwargs.get('batch_dims', '')

    operands = list(operands)
    for i, operand in enumerate(operands):
        raise NotImplementedError('TODO')


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
        backend = kwargs.pop('backend', 'torch')
        if backend == 'torch':
            einsum = torch_einsum
        else:
            einsum = getattr(importlib.import_module(backend), 'einsum')

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
            op_equation = ','.join(op_inputs) + '->' + op_output
            # TODO share intermediates
            tensor = einsum(op_equation, *op_tensors)
            remaining.append((op_output, tensor))
        assert len(remaining) == 1
        return remaining[0][1]
