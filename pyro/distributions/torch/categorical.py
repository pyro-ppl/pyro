from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.categorical import Categorical as _Categorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_Categorical)
class Categorical(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.Categorical(probs=ps, logits=logits)
        x_shape = ps.shape[:-1] + (1,) if ps is not None \
            else logits.shape[:-1] + (1,)
        event_dim = 1
        super(Categorical, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    def sample(self):
        x = self.torch_dist.sample(self._sample_shape)
        return x.view(self._sample_shape + self._x_shape)

    def batch_log_pdf(self, x):
        log_pxs = self.torch_dist.log_prob(x.squeeze(-1)).unsqueeze(-1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        batch_log_pdf = torch.sum(log_pxs, -1).contiguous().view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        values = self.torch_dist.enumerate_support()
        sample_shape = (self.torch_dist.probs.shape[-1],)
        return values.view(sample_shape + self._x_shape)
