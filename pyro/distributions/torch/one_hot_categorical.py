from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.one_hot_categorical import OneHotCategorical as _OneHotCategorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_OneHotCategorical)
class OneHotCategorical(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        if logits is not None:
            ps = torch.exp(logits - torch.max(logits))
            ps /= ps.sum(-1, True)
        torch_dist = torch.distributions.Categorical(ps)  # TODO switch to OneHotCategorical
        x_shape = ps.shape
        event_dim = 1
        super(OneHotCategorical, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
        self._work_around_lack_of_scalar = (ps.dim() == 1 and ps.sum().dim() == 1)

    def sample(self):
        indices = self.torch_dist.sample(self._sample_shape).data
        if not self._work_around_lack_of_scalar:
            indices = indices.unsqueeze(-1)
        ps = self.torch_dist.probs.data
        zero = ps.new(self._sample_shape + ps.shape).zero_()
        one_hot = zero.scatter_(-1, indices, 1)
        return Variable(one_hot)

    def batch_log_pdf(self, x):
        return self.torch_dist.log_prob(x.max(-1)[1])

    def enumerate_support(self):
        ps = self.torch_dist.probs.data
        n = ps.Size(-1)
        values = torch.eye(n, out=ps.new(n, n))
        values = values.view((n,) + (1,) * (ps.dim() - 1) + (n,))
        return values.expand((n,) + ps.shape)
