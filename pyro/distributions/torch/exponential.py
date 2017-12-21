from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.exponential import Exponential as _Exponential
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Exponential)
class Exponential(TorchDistribution):
    reparameterized = True

    def __init__(self, lam, *args, **kwargs):
        torch_dist = torch.distributions.Exponential(lam)
        super(Exponential, self).__init__(torch_dist, *args, **kwargs)
        self._param_shape = torch.Size(broadcast_shape(lam.size(), strict=True))

    def batch_shape(self, x=None):
        x_shape = [] if x is None else x.size()
        shape = torch.Size(broadcast_shape(x_shape, self._param_shape, strict=True))
        return shape[:-1]

    def event_shape(self):
        return self._param_shape[-1:]
