from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.categorical import Categorical as _Categorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Categorical)
class Categorical(TorchDistribution):
    """
    Compatibility wrapper around
    `torch.distributions.Categorical <http://pytorch.org/docs/master/_modules/torch/distributions.html#Categorical>`_
    """
    enumerable = True

    def __init__(self, ps=None, vs=None, logits=None, *args, **kwargs):
        if logits is not None:
            ps = torch.exp(logits - torch.max(logits))
            ps /= ps.sum(-1, True)
        self._param_shape = ps.size()
        torch_dist = torch.distributions.Categorical(ps)
        super(Categorical, self).__init__(torch_dist, *args, **kwargs)

    def batch_shape(self, x=None):
        x_shape = [] if x is None else x.size()
        shape = torch.Size(broadcast_shape(x_shape, self._param_shape, strict=True))
        return shape[:-1]

    def event_shape(self):
        return self._param_shape[-1:]

    def batch_log_pdf(self, x):
        return self.torch_dist.log_prob(x.long())

    def sample(self):
        return self.torch_dist.sample().float()
