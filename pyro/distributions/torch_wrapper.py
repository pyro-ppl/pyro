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

    def __init__(self, torch_dist, log_pdf_mask=None, extra_event_dims=0):
        super(TorchDistribution, self).__init__()
        self.torch_dist = torch_dist
        self.log_pdf_mask = log_pdf_mask
        self.extra_event_dims = extra_event_dims

    def batch_shape(self):
        if self.extra_event_dims == 0:
            return self.torch_dist.batch_shape
        batch_dims = len(self.torch_dist.batch_shape) - self.extra_event_dims
        return self.torch_dist.batch_shape[:batch_dims]

    def event_shape(self):
        if self.extra_event_dims == 0:
            return self.torch_dist.event_shape
        batch_dims = len(self.torch_dist.batch_shape) - self.extra_event_dims
        shape = self.torch_dist.batch_shape + self.torch_dist.event_shape
        return shape[batch_dims:]

    def sample(self, sample_shape=torch.Size()):
        if self.reparameterized:
            return self.torch_dist.rsample(sample_shape)
        else:
            return self.torch_dist.sample(sample_shape)

    def log_prob(self, x):
        log_prob = self.torch_dist.log_prob(x)
        for _ in range(self.extra_event_dims):
            log_prob = log_prob.sum(-1)
        if self.log_pdf_mask is not None:
            # Prevent accidental broadcasting of log_prob tensor,
            # e.g. (64, 1), (64,) --> (64, 64)
            assert len(self.log_pdf_mask.size()) <= len(log_prob.size())
            log_prob = log_prob * self.log_pdf_mask
        return log_prob

    def enumerate_support(self):
        return self.torch_dist.enumerate_support()
