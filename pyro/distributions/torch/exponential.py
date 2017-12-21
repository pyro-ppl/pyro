from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.exponential import Exponential as _Exponential
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_Exponential)
class Exponential(TorchDistribution):
    reparameterized = True

    def __init__(self, lam, *args, **kwargs):
        torch_dist = torch.distributions.Exponential(lam)
        x_shape = lam.size()
        event_dim = 1
        super(Exponential, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
