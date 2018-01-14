from __future__ import absolute_import, division, print_function

import torch
from torch.nn.parameter import Parameter

from .kernel import Kernel
from pyro.util import ng_ones, ng_zeros


class RBF(Kernel):
    def __init__(self, variance=torch.ones(1), lengthscale=torch.ones(1)):
        self.variance = Parameter(variance)
        self.lengthscale = Parameter(lengthscale)

    def forward(self, x, z=None):
        if z is None:
            z = x
        r = torch.abs(x - z)
        return self.variance * torch.exp(0.5 * r**2 / self.lengthscale)
