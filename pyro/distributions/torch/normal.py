from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.normal import Normal as _Normal
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Normal)
class Normal(TorchDistribution):
    reparameterized = True

    def __init__(self, mu, sigma, *args, **kwargs):
        torch_dist = torch.distributions.Normal(mu, sigma)
        x_shape = torch.Size(broadcast_shape(mu.size(), sigma.size(), strict=True))
        event_dim = 1
        super(Normal, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
