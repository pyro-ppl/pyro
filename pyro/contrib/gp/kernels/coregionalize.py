# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam


class Coregionalize(Kernel):
    r"""
    A kernel for the linear model of coregionalization
    :math:`k(x,z) = x^T (W W^T + D) z` where :math:`W` is an
    ``input_dim``-by-``rank`` matrix and typically ``rank < input_dim``,
    and ``D`` is a diagonal matrix.

    This generalizes the
    :class:`~pyro.contrib.gp.kernels.dot_product.Linear` kernel to multiple
    features with a low-rank-plus-diagonal weight matrix. The typical use case
    is for modeling correlations among outputs of a multi-output GP, where
    outputs are coded as distinct data points with one-hot coded features
    denoting which output each datapoint represents.

    If only ``rank`` is specified, the kernel ``(W W^T + D)`` will be
    randomly initialized to a matrix with expected value the identity matrix.

    References:

    [1] Mauricio A. Alvarez, Lorenzo Rosasco, Neil D. Lawrence (2012)
        Kernels for Vector-Valued Functions: a Review

    :param int input_dim: Number of feature dimensions of inputs.
    :param int rank: Optional rank. This is only used if ``components`` is
        unspecified. If neigher ``rank`` nor ``components`` is specified,
        then ``rank`` defaults to ``input_dim``.
    :param torch.Tensor components: An optional ``(input_dim, rank)`` shaped
        matrix that maps features to ``rank``-many components. If unspecified,
        this will be randomly initialized.
    :param torch.Tensor diagonal: An optional vector of length ``input_dim``.
        If unspecified, this will be set to constant ``0.5``.
    :param list active_dims: List of feature dimensions of the input which the
        kernel acts on.
    :param str name: Name of the kernel.
    """

    def __init__(self, input_dim, rank=None, components=None, diagonal=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        # Add a low-rank kernel with expected value torch.eye(input_dim, input_dim) / 2.
        if components is None:
            rank = input_dim if rank is None else rank
            components = torch.randn(input_dim, rank) * (0.5 / rank) ** 0.5
        else:
            rank = components.size(-1)
        if components.shape != (input_dim, rank):
            raise ValueError("Expected components.shape == ({},rank), actual {}"
                             .format(input_dim, components.shape))
        self.components = Parameter(components)

        # Add a diagonal component initialized to torch.eye(input_dim, input_dim) / 2,
        # such that the total kernel has expected value the identity matrix.
        diagonal = components.new_ones(input_dim) * 0.5 if diagonal is None else diagonal
        if diagonal.shape != (input_dim,):
            raise ValueError("Expected diagonal.shape == ({},), actual {}"
                             .format(input_dim, diagonal.shape))
        self.diagonal = PyroParam(diagonal, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        X = self._slice_input(X)
        Xc = X.matmul(self.components)

        if diag:
            return (Xc ** 2).sum(-1) + (X ** 2).mv(self.diagonal)

        if Z is None:
            Z = X
            Zc = Xc
        else:
            Z = self._slice_input(Z)
            Zc = Z.matmul(self.components)

        return Xc.matmul(Zc.t()) + (X * self.diagonal).matmul(Z.t())
