# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam


class Constant(Kernel):
    r"""
    Implementation of Constant kernel:

        :math:`k(x, z) = \sigma^2.`
    """
    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.variance.expand(X.size(0))

        if Z is None:
            Z = X
        return self.variance.expand(X.size(0), Z.size(0))


class WhiteNoise(Kernel):
    r"""
    Implementation of WhiteNoise kernel:

        :math:`k(x, z) = \sigma^2 \delta(x, z),`

    where :math:`\delta` is a Dirac delta function.
    """
    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.variance.expand(X.size(0))

        if Z is None:
            return self.variance.expand(X.size(0)).diag()
        else:
            return X.data.new_zeros(X.size(0), Z.size(0))
