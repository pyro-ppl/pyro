from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.beta import Beta as _Beta
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Beta)
class Beta(TorchDistribution):
    reparameterized = True

    def __init__(self, alpha, beta, *args, **kwargs):
        torch_dist = torch.distributions.Beta(alpha, beta)
        x_shape = torch.Size(broadcast_shape(alpha.size(), beta.size(), strict=True))
        event_dim = 1
        super(Beta, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
