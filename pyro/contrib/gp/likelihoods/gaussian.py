from __future__ import absolute_import, division, print_function

import torch
from torch.nn import Parameter

import pyro.distributions as dist

from .likelihood import Likelihood


class Gaussian(Likelihood):
    """
    Implementation of Gaussian likelihood .

    By default, parameters will be `torch.nn.Parameter`s containing `torch.FloatTensor`s.
    To cast them to the correct data type or GPU device, we can call methods such as
    `.double()`, `.cuda(device=None)`.

    :param torch.Tensor variance: Dimension of inputs for this kernel.
    """

    def __init__(self, variance=None):
        if variance is None:
            variance = torch.ones(1)
        self.variance = Parameter(variance)

    def forward(self, f, obs=None):
        variance = self.variance

        return pyro.sample("y", dist.Normal(f, variance), obs=obs)
