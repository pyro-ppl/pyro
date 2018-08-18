from __future__ import absolute_import, division, print_function

import operator
from numbers import Number

import opt_einsum
import torch
from six.moves import reduce

from pyro.distributions.util import broadcast_shape
from pyro.ops.einsum import contract


def zip_align_right(xs, ys):
    return reversed(list(zip(reversed(xs), reversed(ys))))


def memoized_squeeze(tensor):
    if hasattr(tensor, '_memoized_squeeze'):
        return tensor._memoized_squeeze
    result = tensor.squeeze()
    tensor._memoized_squeeze = result
    return result


def sumproduct(factors, target_shape=(), backend='torch', optimize=True):
    # Handle numbers and trivial cases.
    if backend.endswith('_log'):
        multiply = operator.add
        unit = 0.
    else:
        multiply = operator.mul
        unit = 1.
    numbers = []
    tensors = []
    for t in factors:
        (numbers if isinstance(t, Number) else tensors).append(t)
    if not tensors:
        return reduce(multiply, numbers, unit)
    shape = broadcast_shape(*(t.shape for t in tensors))
    if numbers:
        number_part = reduce(multiply, numbers, unit)
        tensor_part = sumproduct(tensors, target_shape, backend=backend, optimize=optimize)
        return multiply(tensor_part, number_part)

    # Work around opt_einsum interface lack of support for pure broadcasting.
    if len(shape) < len(target_shape) or \
            any(s < t for s, t in zip_align_right(shape, target_shape)):
        smaller_shape = list(target_shape)
        for i in range(len(target_shape)):
            if i >= len(shape) or shape[-1-i] < target_shape[-1-i]:
                smaller_shape[-1-i] = 1
        while smaller_shape and smaller_shape[0] == 1:
            smaller_shape = smaller_shape[1:]
        smaller_shape = tuple(smaller_shape)
        result = sumproduct(factors, smaller_shape, backend=backend, optimize=optimize)
        return result.expand(target_shape)

    if not optimize:
        return naive_sumproduct(tensors, target_shape)
    else:
        return opt_sumproduct(tensors, target_shape, backend=backend)


def naive_sumproduct(factors, target_shape):
    assert all(isinstance(t, torch.Tensor) for t in factors)

    result = factors[0]
    for factor in factors[1:]:
        result = result * factor

    while result.dim() > len(target_shape):
        result = result.sum(0)
    while result.dim() < len(target_shape):
        result = result.unsqueeze(0)
    for dim, (result_size, target_size) in enumerate(zip(result.shape, target_shape)):
        if result_size > target_size:
            result = result.sum(dim, True)

    return result


def opt_sumproduct(factors, target_shape, backend='torch'):
    assert all(isinstance(t, torch.Tensor) for t in factors)
    num_symbols = len(target_shape)
    num_symbols = max(num_symbols, max(len(t.shape) for t in factors))
    symbols = [opt_einsum.get_symbol(i) for i in range(num_symbols)]
    target_names = [name
                    for name, size in zip_align_right(symbols, target_shape)
                    if size != 1]

    # Construct low-dimensional tensors with symbolic names.
    packed_names = []
    packed_factors = []
    for factor in factors:
        packed_names.append([
            name
            for name, size in zip_align_right(symbols, factor.shape)
            if size != 1])
        # memoize the .squeeze() to support shared_intermediates
        packed_factors.append(memoized_squeeze(factor))
        assert len(packed_factors[-1].shape) == len(packed_names[-1])

    # Contract packed tensors.
    expr = '{}->{}'.format(','.join(''.join(names) for names in packed_names),
                           ''.join(target_names))
    packed_result = contract(expr, *packed_factors, backend=backend)

    # Unpack result.
    return packed_result.reshape(target_shape)
