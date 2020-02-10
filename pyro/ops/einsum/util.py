# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0 AND MIT

EINSUM_SYMBOLS_BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class Tensordot:
    """
    Creates a tensordot implementation from an einsum implementation.
    """
    def __init__(self, einsum):
        self.einsum = einsum

    # Copyright (c) 2014 Daniel Smith
    # SPDX-License-Identifier: MIT
    # This function is copied and adapted from:
    # https://github.com/dgasmith/opt_einsum/blob/a6dd686/opt_einsum/backends/torch.py
    def __call__(self, x, y, axes=2):
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
        return self.einsum(einsum_str, x, y)
