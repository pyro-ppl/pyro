# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
from torch.distributions import Transform, constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Transform)
class ConditionedMatrixExponential(Transform):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, weights=None, iterations=8, spectral_norm=0.5):
        super().__init__(cache_size=1)
        assert iterations > 0
        self.weights = weights
        self.iterations = iterations
        self.spectral_norm = spectral_norm

    def _exp(self, x, M):
        """
        Performs power series approximation to the vector product of x with the
        matrix exponential of M.
        """
        power_term = x.unsqueeze(-1)
        y = x.unsqueeze(-1)
        for idx in range(self.iterations):
            power_term = torch.matmul(M, power_term) / (idx + 1)
            y = y + power_term

        return y.squeeze(-1)

    def _trace(self, M):
        """
        Calculate the trace of a matrix and is able to do broadcasting over batch
        dimensions, unlike `torch.trace`.

        Broadcasting is necessary for the conditional version of the transform,
        where `self.weights` may have batch dimensions corresponding the batch
        dimensions of the context variable that was conditioned upon.
        """
        return M.diagonal(dim1=-2, dim2=-1).sum(-1)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x => y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        # TODO: Apply spectral norm
        M = self.weights

        return self._exp(x, M)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x.
        """

        # TODO: Apply spectral norm
        M = self.weights

        return self._exp(y, -M)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """

        # TODO: Apply spectral norm
        M = self.weights

        return self._trace(M)


@copy_docs_from(ConditionedMatrixExponential)
class MatrixExponential(ConditionedMatrixExponential, TransformModule):
    r"""
    A 'planar' bijective transform with equation,

        :math:`\mathbf{y} = \mathbf{x} + \mathbf{u}\tanh(\mathbf{w}^T\mathbf{z}+b)`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    and the learnable parameters are :math:`b\in\mathbb{R}`,
    :math:`\mathbf{u}\in\mathbb{R}^D`, :math:`\mathbf{w}\in\mathbb{R}^D` for
    input dimension :math:`D`. For this to be an invertible transformation, the
    condition :math:`\mathbf{w}^T\mathbf{u}>-1` is enforced.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Planar(10)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int

    References:

    [1] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, input_dim):
        super().__init__()

        self.weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(0))
        self.weights.data.uniform_(-stdv, stdv)


def matrix_exponential(input_dim):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.MatrixExponential` object for consistency
    with other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    return MatrixExponential(input_dim)
