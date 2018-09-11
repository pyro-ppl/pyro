from __future__ import absolute_import, division, print_function

import torch

EINSUM_SYMBOLS_BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def transpose(a, axes):
    return a.permute(*axes)


def einsum(equation, *operands):
    """
    Log-sum-exp implementation of einsum.
    """
    # rename symbols to support PyTorch 0.4.1 and earlier,
    # which allow only symbols a-z.
    symbols = sorted(set(equation) - set(',->'))
    rename = dict(zip(symbols, 'abcdefghijklmnopqrstuvwxyz'))
    equation = ''.join(rename.get(s, s) for s in equation)

    inputs, output = equation.split('->')
    inputs = inputs.split(',')

    shifts = []
    exp_operands = []
    for dims, operand in zip(inputs, operands):
        shift = operand
        for i, dim in enumerate(dims):
            if dim not in output:
                shift = shift.max(i, keepdim=True)[0]
        exp_operands.append((operand - shift).exp())

        # permute shift to match output
        shift = shift.squeeze()
        dims = [dim for dim in dims if dim in output]
        dims = [dim for dim in output if dim not in dims] + dims
        shift = shift.reshape((1,) * (len(output) - len(shift.shape)) + shift.shape)
        if dims:
            shift = shift.permute(*(dims.index(dim) for dim in output))
        shifts.append(shift)

    result = torch.einsum(equation, exp_operands).log()
    return sum(shifts + [result])


# Copyright (c) 2014 Daniel Smith
# This function is copied and adapted from:
# https://github.com/dgasmith/opt_einsum/blob/a6dd686/opt_einsum/backends/torch.py
def tensordot(x, y, axes=2):
    # convert int argument to (list[int], list[int])
    if isinstance(axes, int):
        axes = list(range(x.dim() - axes, x.dim())), list(range(axes))

    # convert (int, int) to (list[int], list[int])
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    # compute shifts
    assert all(dim >= 0 for axis in axes for dim in axis)
    x_shift = x
    y_shift = y
    for dim in axes[0]:
        x_shift = x_shift.max(dim, keepdim=True)[0]
    for dim in axes[1]:
        y_shift = y_shift.max(dim, keepdim=True)[0]

    result = torch.tensordot((x - x_shift).exp(), (y - y_shift).exp(), axes).log()

    # apply shifts to result
    x_part = x.dim() - len(axes[0])
    y_part = y.dim() - len(axes[1])
    assert result.dim() == x_part + y_part
    result += x_shift.reshape(result.shape[:x_part] + (1,) * y_part)
    result += y_shift.reshape(result.shape[x_part:])

    return result
