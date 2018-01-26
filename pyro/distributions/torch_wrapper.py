from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from


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

    def log_prob(self, x):
        log_prob = self.torch_dist.log_prob(x)
        if self.log_pdf_mask is not None:
            log_prob = log_prob * self.log_pdf_mask
        return log_prob

    def enumerate_support(self):
        return self.torch_dist.enumerate_support()
