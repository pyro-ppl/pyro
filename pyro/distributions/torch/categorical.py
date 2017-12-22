from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.categorical import Categorical as _Categorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_Categorical)
class Categorical(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        if logits is not None:
            ps = torch.exp(logits - torch.max(logits))
            ps /= ps.sum(-1, True)
        torch_dist = torch.distributions.Categorical(ps)
        x_shape = ps.shape[:-1] + (1,)
        event_dim = 1
        super(Categorical, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
        self._work_around_lack_of_scalar = (ps.dim() == 1 and ps.sum().dim() == 1)

    def sample(self):
        x = self.torch_dist.sample(self._sample_shape).float()
        return x.view(self._sample_shape + self._x_shape)

    def batch_log_pdf(self, x):
        # if not self._work_around_lack_of_scalar:
        #     x = x.squeeze(-1)
        batch_log_pdf = self.torch_dist.log_prob(x.long())
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        values = self.torch_dist.enumerate_support().float()
        return values.view((values.shape[0],) + self._x_shape)
        # return self.torch_dist.enumerate_support().unsqueeze(-1).float()
