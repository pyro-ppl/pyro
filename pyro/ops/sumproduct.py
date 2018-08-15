from __future__ import absolute_import, division, print_function

import opt_einsum
import torch

from pyro.distributions.util import broadcast_shape


def _product(factors):
    result = 1.
    for factor in factors:
        result = result * factor
    return result


def sumproduct(factors, target_shape, optimize=True):
    # Handle numbers and trivial cases.
    numbers = []
    tensors = []
    for t in factors:
        (tensors if isinstance(t, torch.Tensor) else numbers).append(t)
    if not tensors:
        return _product(numbers)
    if numbers:
        number_part = _product(numbers)
        tensor_part = sumproduct(tensors, target_shape, optimize=optimize)
        shape = broadcast_shape(*(t.shape for t in tensors))
        contracted_shape = shape[:len(shape) - len(target_shape)]
        replication_power = _product(contracted_shape)
        return tensor_part * number_part ** replication_power

    if not optimize:
        return naive_sumproduct(tensors, target_shape)
    else:
        return opt_sumproduct(tensors, target_shape)


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


def opt_sumproduct(factors, target_shape):
    assert all(isinstance(t, torch.Tensor) for t in factors)

    num_symbols = len(target_shape)
    num_symbols = max(num_symbols, max(t.dim() for t in factors))
    symbols = [opt_einsum.get_symbol(i) for i in range(num_symbols)]
    rev_symbols = list(reversed(symbols))
    target_names = [name
                    for name, size in zip(rev_symbols, reversed(target_shape))
                    if size != 1]

    # Construct low-dimensional tensors with symbolic names.
    packed_names = []
    packed_factors = []
    for factor in factors:
        packed_names.append([
            name
            for name, size in zip(rev_symbols, reversed(factor.shape))
            if size != 1])
        packed_factors.append(factor.squeeze().clone())  # FIXME remove this clone
        assert packed_factors[-1].dim() == len(packed_names[-1])

    # Contract packed tensors.
    expr = '{}->{}'.format(','.join(''.join(names) for names in packed_names),
                           ''.join(target_names))
    packed_result = opt_einsum.contract(expr, *packed_factors, backend='torch')

    # Unpack result.
    result = packed_result
    for dim in range(-1, -1 - len(target_shape), -1):
        if target_shape[dim] == 1:
            result = result.unsqueeze(dim)
    assert result.shape == target_shape
    return result
