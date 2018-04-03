from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


class Constant(Kernel):
    """
    Implementation of Constant kernel:

        :math:`k(x, z) = \sigma^2.`
    """
    def __init__(self, input_dim, variance=None, active_dims=None, name="Constant"):
        super(Constant, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")
        if diag:
            return variance.expand(X.shape[0])

        if Z is None:
            Z = X
        return variance.expand(X.shape[0], Z.shape[0])


class WhiteNoise(Kernel):
    """
    Implementation of WhiteNoise kernel:

        :math:`k(x, z) = \sigma^2 \delta(x, z),`

    where :math:`\delta` is a Dirac delta function.
    """
    def __init__(self, input_dim, variance=None, active_dims=None, name="WhiteNoise"):
        super(WhiteNoise, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")
        if diag:
            return variance.expand(X.shape[0])

        if Z is None:
            return variance.expand(X.shape[0]).diag()
        else:
            return X.data.new_zeros(X.shape[0], Z.shape[0])
