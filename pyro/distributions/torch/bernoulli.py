from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F

from pyro.distributions.bernoulli import Bernoulli as _Bernoulli
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_Bernoulli)
class Bernoulli(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        if (ps is None) == (logits is None):
            raise ValueError("Got ps={}, logits={}. Either `ps` or `logits` must be specified, "
                             "but not both.".format(ps, logits))
        if ps is None:
            ps = F.sigmoid(logits)
        torch_dist = torch.distributions.Bernoulli(ps)
        x_shape = ps.size()
        event_dim = 1
        super(Bernoulli, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
