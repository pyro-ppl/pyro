# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam


class DotProduct(Kernel):
    r"""
    Base class for kernels which are functions of :math:`x \cdot z`.
    """

    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def _dot_product(self, X, Z=None, diag=False):
        r"""
        Returns :math:`X \cdot Z`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            return (X ** 2).sum(-1)

        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        return X.matmul(Z.t())


class Linear(DotProduct):
    r"""
    Implementation of Linear kernel:

        :math:`k(x, z) = \sigma^2 x \cdot z.`

    Doing Gaussian Process regression with linear kernel is equivalent to doing a
    linear regression.

    .. note:: Here we implement the homogeneous version. To use the inhomogeneous
        version, consider using :class:`Polynomial` kernel with ``degree=1`` or making
        a :class:`.Sum` with a :class:`.Constant` kernel.
    """

    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, variance, active_dims)

    def forward(self, X, Z=None, diag=False):
        return self.variance * self._dot_product(X, Z, diag)


class Polynomial(DotProduct):
    r"""
    Implementation of Polynomial kernel:

        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`

    :param torch.Tensor bias: Bias parameter of this kernel. Should be positive.
    :param int degree: Degree :math:`d` of the polynomial.
    """

    def __init__(self, input_dim, variance=None, bias=None, degree=1, active_dims=None):
        super().__init__(input_dim, variance, active_dims)

        bias = torch.tensor(1.) if bias is None else bias
        self.bias = PyroParam(bias, constraints.positive)

        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree for Polynomial kernel should be a positive integer.")
        self.degree = degree

    def forward(self, X, Z=None, diag=False):
        return self.variance * ((self.bias + self._dot_product(X, Z, diag)) ** self.degree)
