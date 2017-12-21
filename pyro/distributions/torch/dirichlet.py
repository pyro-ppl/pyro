from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.dirichlet import Dirichlet as _Dirichlet
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_Dirichlet)
class Dirichlet(TorchDistribution):
    reparameterized = True

    def __init__(self, alpha, *args, **kwargs):
        torch_dist = torch.distributions.Dirichlet(alpha)
        x_shape = alpha.size()
        event_dim = 1
        super(Dirichlet, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    def batch_log_pdf(self, x):
        batch_log_pdf = self.torch_dist.log_prob(x).view(self.batch_shape(x) + (1,))
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf
