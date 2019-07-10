from __future__ import absolute_import, division, print_function

from opt_einsum.parser import alpha_canonicalize

import pyro.ops.einsum.torch_log
import pyro.ops.jit


def transpose(a, axes):
    return a.permute(axes)


@pyro.ops.jit.trace()
def _einsum(*args, **kwargs):
    operands = args
    equation = alpha_canonicalize(kwargs["equation"])
    return pyro.ops.einsum.torch_log.einsum(equation, *operands)


def einsum(equation, *operands):
    return _einsum(*operands, equation=equation)


@pyro.ops.jit.trace()
def _tensordot(x, y, **kwargs):
    axes = kwargs["axes"]
    return pyro.ops.einsum.torch_log.tensordot(x, y, axes=axes)


def tensordot(x, y, axes=2):
    if isinstance(axes, list):
        axes = tuple(axes)
    return _tensordot(x, y, axes=axes, x_dim=x.dim(), y_dim=y.dim())
