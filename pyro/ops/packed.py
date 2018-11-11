from __future__ import absolute_import, division, print_function

import math
import operator

import torch
from six.moves import reduce

from pyro.distributions.util import is_identically_one
from pyro.ops.einsum import contract


def pack(value, dim_to_symbol):
    """
    Converts an unpacked tensor to a packed tensor.

    :param value: a number or tensor
    :param dim_to_symbol: a map from negative integers to characters
    """
    if isinstance(value, torch.Tensor):
        assert not hasattr(value, '_pyro_dims'), 'tried to pack an already-packed tensor'
        shape = value.shape
        shift = len(shape)
        dims = ''.join(dim_to_symbol[dim - shift] for dim, size in enumerate(shape) if size > 1)
        value = value.squeeze()
        value._pyro_dims = dims
    return value


def unpack(value, symbol_to_dim):
    """
    Converts a packed tensor to an unpacked tensor.

    :param value: a number or tensor
    :param symbol_to_dim: a map from characters to negative integers
    """
    if isinstance(value, torch.Tensor):
        dims = [symbol_to_dim[dim] for dim in value._pyro_dims]
        shape = [1] * -min(dims) if dims else []
        for dim, size in zip(dims, value.shape):
            shape[dim] = size
        value = value.reshape(shape)
    return value


def broadcast_all(*values):
    """
    Packed broadcasting of multiple tensors.
    """
    sizes = {dim: size for value in values for dim, size in zip(value._pyro_dims, value.shape)}
    dims = ''.join(sorted(sizes))
    shape = torch.Size(sizes[dim] for dim in dims)
    values = list(values)
    for i, x in enumerate(values):
        old_dims = x._pyro_dims
        if old_dims != dims:
            x = x.permute(tuple(old_dims.index(dim) for dim in dims if dim in old_dims))
            x = x.reshape(tuple(sizes[dim] if dim in old_dims else 1 for dim in dims))
            x = x.expand(shape)
            x._pyro_dims = dims
            values[i] = x
    return tuple(values)


def gather(value, index, dim):
    """
    Packed broadcasted gather of indexed values along a named dim.
    """
    assert dim in value._pyro_dims
    assert dim not in index._pyro_dims
    value, index = broadcast_all(value, index)
    dims = value._pyro_dims.replace(dim, '')
    index = index.index_select(index._pyro_dims.index(dim),
                               index.new_tensor([0], dtype=torch.long))
    value = value.gather(dim, index.long())
    value._pyro_dims = dims
    return value


def mul(lhs, rhs):
    """
    Packed broadcasted multiplication.
    """
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        dims = ''.join(sorted(set(lhs._pyro_dims + rhs._pyro_dims)))
        equation = lhs._pyro_dims + ',' + rhs._pyro_dims + '->' + dims
        result = torch.einsum(equation, lhs, rhs, backend='torch')
        result._pyro_dims = dims
        return result
    result = lhs * rhs
    if isinstance(lhs, torch.Tensor):
        result._pyro_dims = lhs._pyro_dims
    elif isinstance(rhs, torch.Tensor):
        result._pyro_dims = rhs._pyro_dims
    return result


def scale_and_mask(tensor, scale=1.0, mask=None):
    """
    Scale and mask a packed tensor, broadcasting and avoiding unnecessary ops.

    :param torch.Tensor tensor: a packed tensor
    :param scale: a positive scale
    :type scale: torch.Tensor or number
    :param mask: an optional packed tensor mask
    :type mask: torch.ByteTensor or None
    """
    if isinstance(scale, torch.Tensor) and scale.dim():
        raise NotImplementedError('non-scalar scale is not supported')
    if mask is None:
        if is_identically_one(scale):
            return tensor
        result = tensor * scale
        result._pyro_dims = tensor._pyro_dims
        return result
    tensor, mask = broadcast_all(tensor, mask)
    result = tensor * scale  # triggers a copy, avoiding in-place op errors
    result.masked_fill_(~mask, 0.)
    result._pyro_dims = tensor._pyro_dims
    return result


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


def sumproduct(factors, output_dims='', device=None):
    """
    Packed sum-product contraction.
    """
    numbers = []
    tensors = []
    for x in factors:
        (tensors if isinstance(x, torch.Tensor) else numbers).append(x)
    if tensors:
        equation = ','.join(x._pyro_dims for x in tensors) + '->' + output_dims
        result = contract(equation, *tensors, backend='torch')
        if numbers:
            result = result * reduce(operator.mul, numbers, 1.)
        result._pyro_dims = output_dims
        return result
    result = torch.tensor(reduce(operator.mul, numbers, 1.), device=device)
    result._pyro_dims = ''
    return result


def logsumproductexp(factors, output_dims='', device=None):
    """
    Packed sum-product contraction in log space.
    """
    numbers = []
    tensors = []
    for x in factors:
        (tensors if isinstance(x, torch.Tensor) else numbers).append(x)
    if tensors:
        equation = ','.join(x._pyro_dims for x in tensors) + '->' + output_dims
        result = contract(equation, *tensors, backend='pyro.ops.einsum.torch_log')
        if numbers:
            result = result + float(sum(numbers))
        result._pyro_dims = output_dims
        return result
    result = torch.tensor(float(sum(numbers)), device=device)
    result._pyro_dims = ''
    return result
