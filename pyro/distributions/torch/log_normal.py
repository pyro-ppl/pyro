from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class LogNormal(TorchDistribution):
    """
    Log-normal distribution.

    The distribution of a random variable whose logarithm is normally
    distributed, i.e. ``ln(X) ~ Normal({mu: mu, sigma: sigma})``.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `sigma` parameters are positive.

    :param torch.autograd.Variable mu: log mean parameter.
    :param torch.autograd.Variable sigma: log standard deviations.
        Should be positive.
    """
    reparameterized = True

    def __init__(self, mu, sigma, *args, **kwargs):
        self.mu = mu
        self.sigma = sigma
        torch_dist = torch.distributions.LogNormal(mu, sigma)
        super(LogNormal, self).__init__(torch_dist, *args, **kwargs)

    def analytic_mean(self):
        return torch.exp(self.mu + 0.5 * torch.pow(self.sigma, 2.0))

    def analytic_var(self):
        return (torch.exp(torch.pow(self.sigma, 2.0)) - Variable(torch.ones(1))) * \
               torch.pow(self.analytic_mean(), 2)
