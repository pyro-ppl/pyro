from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(TorchDistribution)
class Normal(TorchDistribution):
    """
    Univariate normal (Gaussian) distribution.

    A distribution over tensors in which each element is independent and
    Gaussian distributed, with its own mean and standard deviation. The
    distribution is over tensors that have the same shape as the parameters `mu`
    and `sigma`, which in turn must have the same shape as each other.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `sigma` parameters are positive.

    :param torch.autograd.Variable mu: Means.
    :param torch.autograd.Variable sigma: Standard deviations.
        Should be positive and the same shape as `mu`.
    """
    reparameterized = True

    def __init__(self, mu, sigma, *args, **kwargs):
        torch_dist = torch.distributions.Normal(mu, sigma)
        x_shape = torch.Size(broadcast_shape(mu.size(), sigma.size(), strict=True))
        event_dim = 1
        super(Normal, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
