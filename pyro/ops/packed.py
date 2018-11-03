from __future__ import absolute_import, division, print_function

import math
import operator

import torch

from pyro.ops.einsum import contract


def pack(value, dim_to_symbol):
    """
    Converts an unpacked tensor to a packed tensor.
    """
    if isinstance(value, torch.Tensor):
        assert not hasattr(value, '_pyro_dims'), 'tried to pack an already-packed tensor'
        shape = value.shape
        shift = len(shape)
        dims = ''.join(dim_to_symbol[dim - shift] for dim, size in enumerate(shape) if size > 1)
        value = value.squeeze()
        value._pyro_dims = dims
    return value


def broadcast_all(*values):
    raise NotImplementedError('TODO')


def neg(value):
    """
    Packed negation.
    """
    result = -value
    if isinstance(value, torch.Tensor):
        result._pyro_dims = value._pyro_dims
    return result


def exp(value):
    """
    Packed pointwise exponential.
    """
    if isinstance(value, torch.Tensor):
        result = value.exp()
        result._pyro_dims = value._pyro_dims
    else:
        result = math.exp(value)
    return result


def mul(lhs, rhs):
    """
    Packed broadcasted multiplication.
    """
    if isinstance(lhs, torch.Tensor):
        if isinstance(rhs, torch.Tensor):
            dims = ''.join(sorted(set(lhs._pyro_dims + rhs._pyro_dims)))
            equation = lhs._pyro_dims + ',' + rhs._pyro_dims + '->' + dims
            result = torch.einsum(equation, lhs._pyro_packed, rhs._pyro_packed,
                                  backend='torch')
            result._pyro_dims = dims
            return result
        result = lhs * rhs
        result._pyro_dims = lhs._pyro_dims
        return result
    result = lhs * rhs
    if isinstance(rhs, torch.Tensor):
        result._pyro_dims = lhs._pyro_dims
    return result


def sumproduct(factors, output_dims):
    """
    Packed sum-product contraction.
    """
    numbers = []
    tensors = []
    for x in factors:
        (tensors if isinstance(x, torch.Tensor) else numbers).append(x)
    if tensors:
        equation = ','.join(x._pyro_dim for x in tensors) + '->' + output_dims
        result = contract(equation, *tensors, backend='torch')
        if numbers:
            result = result * reduce(operator.mul, numbers, 1.)
        result._pyro_dims = output_dims
    return reduce(operator.mul, numbers, 1.)


def logsumproductexp(factors, output_dims):
    """
    Packed sum-product contraction in log space.
    """
    numbers = []
    tensors = []
    for x in factors:
        (tensors if isinstance(x, torch.Tensor) else numbers).append(x)
    if tensors:
        equation = ','.join(x._pyro_dim for x in tensors) + '->' + output_dims
        result = contract(equation, *tensors, backend='pyro.ops.einsum.torch_log')
        if numbers:
            result = result + reduce(operator.add, numbers, 0.)
        result._pyro_dims = output_dims
    return reduce(operator.add, numbers, 0.)
