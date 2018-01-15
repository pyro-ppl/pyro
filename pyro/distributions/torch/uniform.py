from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.uniform import Uniform as _Uniform
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Uniform)
class Uniform(TorchDistribution):
    reparameterized = True

    def __init__(self, a, b, *args, **kwargs):
        torch_dist = torch.distributions.Uniform(a, b)
        x_shape = torch.Size(broadcast_shape(a.size(), b.size(), strict=True))
        event_dim = 1
        super(Uniform, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
