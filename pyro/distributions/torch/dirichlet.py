from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.dirichlet import Dirichlet as _Dirichlet
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Dirichlet)
class Dirichlet(TorchDistribution):
    reparameterized = True

    def __init__(self, alpha, *args, **kwargs):
        torch_dist = torch.distributions.Dirichlet(alpha)
        super(Dirichlet, self).__init__(torch_dist, *args, **kwargs)
        self._param_shape = alpha.size()

    def batch_shape(self, x=None):
        x_shape = [] if x is None else x.size()
        shape = torch.Size(broadcast_shape(x_shape, self._param_shape, strict=True))
        return shape[:-1]

    def event_shape(self):
        return self._param_shape[-1:]

    def batch_log_pdf(self, x):
        batch_log_pdf = self.torch_dist.log_prob(x).view(self.batch_shape(x) + (1,))
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf
