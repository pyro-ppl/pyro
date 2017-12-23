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

    def __init__(self, torch_dist, x_shape, event_dim, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        super(Distribution, self).__init__(*args, **kwargs)
        self.torch_dist = torch_dist
        self.log_pdf_mask = log_pdf_mask
        self._x_shape = x_shape
        self._event_dim = event_dim
        self._sample_shape = torch.Size() if batch_size is None else torch.Size((batch_size,))

    def batch_shape(self, x=None):
        x_shape = [] if x is None else x.size()
        shape = torch.Size(broadcast_shape(x_shape, self._x_shape, strict=True))
        event_start = len(shape) - self._event_dim
        return shape[:event_start]

    def event_shape(self):
        event_start = len(self._x_shape) - self._event_dim
        return self._x_shape[event_start:]

    def sample(self):
        if self.reparameterized:
            return self.torch_dist.rsample(self._sample_shape)
        else:
            return self.torch_dist.sample(self._sample_shape)

    def batch_log_pdf(self, x):
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_pxs = self.torch_dist.log_prob(x)
        batch_log_pdf = torch.sum(log_pxs, -1).contiguous().view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        return self.torch_dist.enumerate_support()
