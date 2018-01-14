from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.one_hot_categorical import OneHotCategorical as _OneHotCategorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_OneHotCategorical)
class OneHotCategorical(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.OneHotCategorical(probs=ps, logits=logits)
        x_shape = ps.shape if ps is not None else logits.shape
        event_dim = 1
        super(OneHotCategorical, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    def batch_log_pdf(self, x):
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_pxs = self.torch_dist.log_prob(x)
        batch_log_pdf = log_pxs.view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        values = self.torch_dist.enumerate_support()
        return values.view(self.event_shape() + self._x_shape)
