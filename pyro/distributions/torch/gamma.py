from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.gamma import Gamma as _Gamma
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Gamma)
class Gamma(TorchDistribution):
    reparameterized = True

    def __init__(self, alpha, beta, *args, **kwargs):
        torch_dist = torch.distributions.Gamma(alpha, beta)
        x_shape = torch.Size(broadcast_shape(alpha.size(), beta.size(), strict=True))
        event_dim = 1
        super(Gamma, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
