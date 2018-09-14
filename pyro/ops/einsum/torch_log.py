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
    xnd = x.ndimension()
    ynd = y.ndimension()

    # convert int argument to (list[int], list[int])
    if isinstance(axes, int):
        axes = range(xnd - axes, xnd), range(axes)

    # convert (int, int) to (list[int], list[int])
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    # initialize empty indices
    x_ix = [None] * xnd
    y_ix = [None] * ynd
    out_ix = []

    # fill in repeated indices
    available_ix = iter(EINSUM_SYMBOLS_BASE)
    for ax1, ax2 in zip(*axes):
        repeat = next(available_ix)
        x_ix[ax1] = repeat
        y_ix[ax2] = repeat

    # fill in the rest, and maintain output order
    for i in range(xnd):
        if x_ix[i] is None:
            leave = next(available_ix)
            x_ix[i] = leave
            out_ix.append(leave)
    for i in range(ynd):
        if y_ix[i] is None:
            leave = next(available_ix)
            y_ix[i] = leave
            out_ix.append(leave)

    # form full string and contract!
    einsum_str = "{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix)))
    return einsum(einsum_str, x, y)
