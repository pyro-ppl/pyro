from __future__ import absolute_import, division, print_function

import torch

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
        torch_dist = torch.distributions.LogNormal(mu, sigma)
        super(LogNormal, self).__init__(torch_dist, *args, **kwargs)
