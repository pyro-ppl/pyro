from __future__ import absolute_import, division, print_function

import torch
from torch.nn import Parameter

from .kernel import Kernel


class RBF(Kernel):
    """
    Implementation of Radial Basis Function kernel.
    """

    def __init__(self, variance=torch.ones(1), lengthscale=torch.ones(1), active_dims=None, name="RBF"):
        super(RBF, self).__init__(active_dims=active_dims, name=name)
        self.variance = Parameter(variance)
        self.lengthscale = Parameter(lengthscale)

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        X = self._slice_X(X)
        Z = self._slice_X(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        X2 = (X ** 2).sum(1, keepdim=True)
        Z2 = (Z ** 2).sum(1, keepdim=True)
        XZ = X.matmul(Z.t())
        d2 = X2 - 2 * XZ + Z2.t()
        return self.variance * torch.exp(-0.5 * d2 / self.lengthscale**2)
