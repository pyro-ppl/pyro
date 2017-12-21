from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.categorical import Categorical as _Categorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_Categorical)
class Categorical(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, vs=None, logits=None, *args, **kwargs):
        if vs is not None:
            raise NotImplementedError
        if logits is not None:
            ps = torch.exp(logits - torch.max(logits))
            ps /= ps.sum(-1, True)
        torch_dist = torch.distributions.Categorical(ps)
        x_shape = ps.size()[:-1]
        event_dim = 0
        super(Categorical, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    def batch_log_pdf(self, x):
        return self.torch_dist.log_prob(x.long())

    def sample(self):
        return self.torch_dist.sample().float()
