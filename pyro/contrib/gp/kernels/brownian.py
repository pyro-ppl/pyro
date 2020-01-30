# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam


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

    def __init__(self, input_dim, variance=None, active_dims=None):
        if input_dim != 1:
            raise ValueError("Input dimensional for Brownian kernel must be 1.")
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            return self.variance * X.abs().squeeze(1)

        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        Zt = Z.t()
        return torch.where(X.sign() == Zt.sign(),
                           self.variance * torch.min(X.abs(), Zt.abs()),
                           X.data.new_zeros(X.size(0), Z.size(0)))
