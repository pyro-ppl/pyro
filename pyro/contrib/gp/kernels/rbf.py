from __future__ import absolute_import, division, print_function

import torch
from torch.nn import Parameter

from .kernel import Kernel


class RBF(Kernel):
    """
    Implementation of RBF kernel.
    """

    def __init__(self, variance=torch.ones(1), lengthscale=torch.ones(1), active_dims=None, name="RBF"):
        super(RBF, self).__init__(active_dims=active_dims, name=name)
        self.variance = Parameter(variance)
        self.lengthscale = Parameter(lengthscale)

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        if X.size() != Z.size():
            raise ValueError("Inputs must have the same shapes.")
        if X.dim() == 1:
            X = X.unsqueeze(1)
            Z = Z.unsqueeze(1)

        X2 = (X ** 2).sum(1, keepdim=True)
        Z2 = (Z ** 2).sum(1, keepdim=True)
        XZ = X.matmul(Z.t())
        d2 = X2 - 2 * XZ + Z2.t()
        return self.variance * torch.exp(-0.5 * d2 / self.lengthscale**2)
