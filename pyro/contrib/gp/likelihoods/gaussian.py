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

    :param torch.Tensor variance: Variance parameter.
    """
    def __init__(self, variance=None, name="Gaussian"):
        super(Gaussian, self).__init__(name)
        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def forward(self, f_loc, f_var, y):
        variance = self.get_param("variance")
        y_var = f_var + variance
        return pyro.sample(self.y_name,
                           dist.Normal(f_loc, y_var)
                               .reshape(sample_shape=y.shape[:-f_loc.dim()],
                                        extra_event_dims=y.dim()),
                           obs=y)
