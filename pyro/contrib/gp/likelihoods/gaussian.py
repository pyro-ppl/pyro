from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


class Gaussian(Likelihood):
    """
    Implementation of Gaussian likelihood, which is used for regression problems.

    Gaussian likelihood uses :class:`~pyro.distributions.Normal` distribution.

    :param torch.Tensor variance: A variance parameter, which plays the role of
        ``noise`` in regression problems.
    """
    def __init__(self, variance=None, name="Gaussian"):
        super(Gaussian, self).__init__(name)
        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

    def forward(self, f_loc, f_var, y=None):
        """
        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}` according to

            .. math:: y \sim \mathbb{Normal}(f_{loc}, f_{var} + \epsilon),

        where :math:`\epsilon` is the ``variance`` parameter of this likelihood.

        :param torch.Tensor f_loc: Mean of latent function output.
        :param torch.Tensor f_var: Variance of latent function output.
        :param torch.Tensor y: Training output tensor.
        :returns: a tensor sampled from likelihood
        :rtype: torch.Tensor
        """
        variance = self.get_param("variance")
        y_var = f_var + variance

        y_dist = dist.Normal(f_loc, y_var)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).independent(y.dim())
        return pyro.sample(self.y_name, y_dist, obs=y)
