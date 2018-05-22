from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


class DotProduct(Kernel):
    """
    Base class for kernels which are functions of :math:`x \cdot z`.
    """

    def __init__(self, input_dim, variance=None, active_dims=None, name=None):
        super(DotProduct, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def _dot_product(self, X, Z=None, diag=False):
        """
        Returns :math:`X \cdot Z`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            return (X ** 2).sum(-1)

        Z = self._slice_input(Z)
        if X.shape[1] != Z.shape[1]:
            raise ValueError("Inputs must have the same number of features.")

        return X.matmul(Z.t())


class Linear(DotProduct):
    """
    Implementation of Linear kernel:

        :math:`k(x, z) = \sigma^2 x \cdot z.`

    Doing Gaussian Process regression with linear kernel is equivalent to doing a
    linear regression.

    .. note:: Here we implement the homogeneous version. To use the inhomogeneous
        version, consider using :class:`Polynomial` kernel with ``degree=1`` or making
        a :class:`.Sum` with a :class:`.Bias` kernel.
    """

    def __init__(self, input_dim, variance=None, active_dims=None, name="Linear"):
        super(Linear, self).__init__(input_dim, variance, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")
        return variance * self._dot_product(X, Z, diag)


class Polynomial(DotProduct):
    r"""
    Implementation of Polynomial kernel:

        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`

    :param torch.Tensor bias: Bias parameter of this kernel. Should be positive.
    :param int degree: Degree :math:`d` of the polynomial.
    """

    def __init__(self, input_dim, variance=None, bias=None, degree=1, active_dims=None,
                 name="Polynomial"):
        super(Polynomial, self).__init__(input_dim, variance, active_dims, name)

        if bias is None:
            bias = torch.tensor(1.)
        self.bias = Parameter(bias)
        self.set_constraint("bias", constraints.positive)

        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree for Polynomial kernel should be a positive "
                             "integer.")
        self.degree = degree

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")
        bias = self.get_param("bias")
        return variance * ((bias + self._dot_product(X, Z, diag)) ** self.degree)
