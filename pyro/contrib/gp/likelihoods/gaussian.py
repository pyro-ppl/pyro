from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


class Gaussian(Likelihood):
    """
    Implementation of Gaussian likelihood.

    :param torch.Tensor variance: Dimension of inputs for this kernel.
    """

    def __init__(self, variance=None):
        super(Gaussian, self).__init__()
        if variance is None:
            variance = torch.ones(1)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def forward(self, f, obs=None):
        variance = self.get_param("variance").expand_as(f)

        return pyro.sample("y", dist.Normal(f, variance), obs=obs)
