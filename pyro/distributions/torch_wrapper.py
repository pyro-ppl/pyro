from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution


class TorchDistribution(Distribution):
    """
    Compatibility wrapper around
    `torch.distributions.Distribution <http://pytorch.org/docs/master/_modules/torch/distributions.html#Distribution>`_
    """

    def __init__(self, torch_dist, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        super(Distribution, self).__init__(*args, **kwargs)
        self.torch_dist = torch_dist
        self.log_pdf_mask = log_pdf_mask
        self.sample_shape = torch.Size() if batch_size is None else torch.Size((batch_size,))

    def sample(self):
        if self.reparameterized:
            return self.torch_dist.rsample(self.sample_shape)
        else:
            return self.torch_dist.sample(self.sample_shape)

    def batch_log_pdf(self, x):
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_pxs = self.torch_dist.log_prob(x)
        batch_log_pdf = torch.sum(log_pxs, -1).contiguous().view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        return self.torch_dist.enumerate_support()
