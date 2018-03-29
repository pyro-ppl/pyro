from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


class Constant(Kernel):
    """
    Implementation of Constant kernel.

    :param torch.Tensor variance: Variance parameter of this kernel.
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
            return variance.expand(X.size(0))

        if Z is None:
            Z = X
        return variance.expand(X.size(0), Z.size(0))


class Bias(Constant):
    """
    Another name of :class:`Constant` kernel.
    """
    def __init__(self, input_dim, variance=None, active_dims=None, name="Bias"):
        super(Bias, self).__init__(input_dim, variance, active_dims, name)


class WhiteNoise(Kernel):
    """
    Implementation of WhiteNoise kernel.

    :param torch.Tensor variance: Variance parameter of this kernel.
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
            return variance.expand(X.size(0))

        if Z is None:
            return variance.expand(X.size(0)).diag()
        else:
            return X.data.new(X.size(0), Z.size(0)).zero_()
