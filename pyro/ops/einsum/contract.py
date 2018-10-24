from __future__ import absolute_import, division, print_function

import importlib
from collections import Counter

import torch

from pyro.ops.einsum.paths import optimize


def torch_einsum(equation, *operands):
    return torch.einsum(equation, operands)


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

        tensors = list(operands)
        for op in self.path:
            op_inputs = []
            op_tensors = []
            for i in reversed(sorted(op)):
                op_tensors.append(tensors.pop(i))
                op_inputs.append(inputs.pop(i))
                ref_counts.subtract(op_inputs[-1])
            op_output = ''.join(sorted(d for d in set(op_inputs) if ref_counts[d]))
            op_equation = ','.join(op_inputs) + '->' + op_output
            # TODO share intermediates
            tensors.append(einsum(op_equation, *op_tensors))
            inputs.append(op_output if inputs else output)
        assert len(tensors) == 1

        return tensors[0]
