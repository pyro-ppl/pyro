from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(Distribution)
class TorchDistribution(Distribution):
    """
    Compatibility wrapper around
    `torch.distributions.Distribution <http://pytorch.org/docs/master/_modules/torch/distributions.html#Distribution>`_
    """

    def __init__(self, torch_dist, log_pdf_mask=None):
        super(TorchDistribution, self).__init__()
        self.torch_dist = torch_dist
        self.log_pdf_mask = log_pdf_mask

    def batch_shape(self):
        return self.torch_dist.batch_shape

    def event_shape(self):
        return self.torch_dist.event_shape

    def sample(self, sample_shape=torch.Size()):
        if self.reparameterized:
            return self.torch_dist.rsample(sample_shape)
        else:
            return self.torch_dist.sample(sample_shape)

    def batch_log_pdf(self, x):
        shape = broadcast_shape(self.shape(), x.size(), strict=True)
        batch_log_pdf_shape = shape[:-1] + (1,)
        log_pxs = self.torch_dist.log_prob(x)
        if len(shape) > len(log_pxs.size()):
            log_pxs = log_pxs.unsqueeze(-1)
        batch_log_pdf = torch.sum(log_pxs, -1).contiguous().view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        return self.torch_dist.enumerate_support()
