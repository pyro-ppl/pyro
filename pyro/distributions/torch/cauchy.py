from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.cauchy import Cauchy as _Cauchy
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Cauchy)
class Cauchy(TorchDistribution):
    reparameterized = True

    def __init__(self, mu, gamma, *args, **kwargs):
        torch_dist = torch.distributions.Cauchy(mu, gamma)
        x_shape = torch.Size(broadcast_shape(mu.size(), gamma.size(), strict=True))
        event_dim = 1
        super(Cauchy, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
