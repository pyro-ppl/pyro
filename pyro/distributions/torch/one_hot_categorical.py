from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.one_hot_categorical import OneHotCategorical as _OneHotCategorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from, get_clamped_probs


@copy_docs_from(_OneHotCategorical)
class OneHotCategorical(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        ps = get_clamped_probs(ps, logits, is_multidimensional=True)
        torch_dist = torch.distributions.Categorical(ps)  # TODO switch to OneHotCategorical
        x_shape = ps.shape
        event_dim = 1
        super(OneHotCategorical, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    def sample(self):
        ps = self.torch_dist.probs.data
        zero = ps.new(self._sample_shape + ps.shape).zero_()
        indices = super(OneHotCategorical, self).sample().data
        if indices.dim() < zero.dim():
            indices = indices.unsqueeze(-1)
        one_hot = zero.scatter_(-1, indices, 1)
        return Variable(one_hot)

    def batch_log_pdf(self, x):
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_pxs = self.torch_dist.log_prob(x.max(-1)[1])
        batch_log_pdf = log_pxs.view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        ps = self.torch_dist.probs.data
        n = ps.shape[-1]
        values = torch.eye(n, out=ps.new(n, n))
        values = values.view((n,) + (1,) * (ps.dim() - 1) + (n,))
        return Variable(values.expand((n,) + ps.shape))
