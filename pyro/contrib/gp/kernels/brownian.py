from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


class Brownian(Kernel):
    r"""
    This kernel correponds to a two-sided Brownion motion (Wiener process):

        :math:`k(x,z)=\begin{cases}\sigma^2\min(|x|,|z|),& \text{if } x\cdot z\ge 0\\
        0, & \text{otherwise}. \end{cases}`

    Note that the input dimension of this kernel must be 1.

    Reference:

    [1] `Theory and Statistical Applications of Stochastic Processes`,
    Yuliya Mishura, Georgiy Shevchenko
    """

    def __init__(self, input_dim, variance=None, active_dims=None, name="Brownian"):
        if input_dim != 1:
            raise ValueError("Input dimensional for Brownian kernel must be 1.")
        super(Brownian, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")

        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            return variance * X.abs().squeeze(1)

        Z = self._slice_input(Z)
        if X.shape[1] != Z.shape[1]:
            raise ValueError("Inputs must have the same number of features.")

        Zt = Z.t()
        return torch.where(X.sign() == Zt.sign(),
                           variance * torch.min(X.abs(), Zt.abs()),
                           X.data.new_zeros(X.shape[0], Z.shape[0]))
