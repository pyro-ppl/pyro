from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


class RBF(Kernel):
    """
    Implementation of Radial Basis Function kernel.

    :param int input_dim: Dimension of inputs for this kernel.
    :param torch.Tensor variance: Variance parameter of this kernel.
    :param torch.Tensor lengthscale: Length scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="RBF"):
        super(RBF, self).__init__(input_dim, active_dims, name)
        if variance is None:
            variance = torch.ones(1)
        self.variance = Parameter(variance)
        if lengthscale is None:
            lengthscale = torch.ones(1)
        lengthscale = lengthscale.expand(self.input_dim).clone()
        self.lengthscale = Parameter(lengthscale)
        self.set_constraint("variance", constraints.positive)
        self.set_constraint("lengthscale", constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.Kdiag(X)
        if Z is None:
            Z = X
        X = self._slice_X(X)
        Z = self._slice_X(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        variance = self.get_param("variance")
        lengthscale = self.get_param("lengthscale")

        scaled_X = X / lengthscale
        scaled_Z = Z / lengthscale
        X2 = (scaled_X ** 2).sum(1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        d2 = X2 - 2 * XZ + Z2.t()
        return variance * torch.exp(-0.5 * d2)

    def Kdiag(self, X):
        variance = self.get_param("variance")
        return variance.expand(X.size(0))
