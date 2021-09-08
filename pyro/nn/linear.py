# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Set

import torch


class FlatBatchLinear(torch.nn.Module):
    """
    Like :class:`~torch.nn.Linear` but with with parameters shared along a
    specified set of internal dimensions.

    Like :class:`~torch.nn.Linear` , this inputs and outputs flat vectors
    (possibly batched). However unlike :class:`~torch.nn.Linear`, this
    internally reshapes the flat input to :param:`shape_in`, and assumes
    weights are constant across input dimensions specified in
    :param:`batch_dims_in`, corresponding to output dimensions
    :param:`batch_dims_out`. The result is flattened from :param:`shape_out` to
    a one-dimensional vector (possibly batched).

    :param torch.Size shape_in: The internal shape of the input tensor.
    :param torch.Size shape_out: The internal shape of the output tensor.
    :param set batch_dims_in: The set of (nonnegative integer) input dims over
        which weights are shared.
    :param set batch_dims_out: The set of (nonnegative integer) output dims
        over which weights are shared.
    """

    def __init__(
        self,
        shape_in: torch.Size,
        shape_out: torch.Size,
        batch_dims_in: Set[int],
        batch_dims_out: Set[int],
    ):
        assert all(isinstance(i, int) and i >= 0 for i in batch_dims_in)
        assert all(isinstance(i, int) and i >= 0 for i in batch_dims_out)
        assert len(batch_dims_in) == len(batch_dims_out)
        for i, j in zip(sorted(batch_dims_in), sorted(batch_dims_out)):
            assert shape_in[i] == shape_out[j]
        super().__init__()

        einsum_in = ""
        einsum_out = ""
        einsum_weight = ""
        shape_weight = []

        # Collect sizes and symbols from the input shape.
        symbols_out = []
        symbols = iter("abcdefghijklmnopqrstuvwxyz")
        for i, size in enumerate(shape_in):
            symbol = next(symbols)
            einsum_in += symbol
            if i in batch_dims_in:
                symbols_out.append(symbol)
            else:
                einsum_weight += symbol
                shape_weight.append(size)

        # Collect sizes and symbols from the output shape.
        symbols_out = iter(symbols_out)
        for i, size in enumerate(shape_out):
            if i in batch_dims_out:
                symbol = next(symbols_out)
            else:
                symbol = next(symbols)
                einsum_weight += symbol
                shape_weight.append(size)
            einsum_out += symbol

        self.einsum_spec = f"...{einsum_in},{einsum_weight}->...{einsum_out}"
        self.weight = torch.nn.Parameter(torch.zeros(shape_weight))
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.batch_dims_in = frozenset(batch_dims_in)
        self.batch_dims_out = frozenset(batch_dims_out)

    def forward(self, flat_in: torch.Tensor) -> torch.Tensor:
        sample_shape = flat_in.shape[:-1]
        shaped_in = flat_in.reshape(sample_shape + self.shape_in)
        shaped_out = torch.einsum(self.einsum_spec, shaped_in, self.weight)
        flat_out = shaped_out.reshape(sample_shape + (-1,))
        return flat_out
