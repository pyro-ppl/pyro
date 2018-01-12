from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.bernoulli import Bernoulli as _Bernoulli
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from, get_clamped_probs


@copy_docs_from(_Bernoulli)
class Bernoulli(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        ps = get_clamped_probs(ps, logits, is_multidimensional=False)
        torch_dist = torch.distributions.Bernoulli(ps)
        x_shape = ps.size()
        event_dim = 1
        super(Bernoulli, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
