from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(TorchDistribution)
class Uniform(TorchDistribution):
    """
    Uniform distribution over the continuous interval `[a, b]`.

    :param torch.autograd.Variable a: lower bound (real).
    :param torch.autograd.Variable b: upper bound (real).
        Should be greater than `a`.
    """
    reparameterized = True

    def __init__(self, a, b, *args, **kwargs):
        torch_dist = torch.distributions.Uniform(a, b)
        x_shape = torch.Size(broadcast_shape(a.size(), b.size(), strict=True))
        event_dim = 1
        super(Uniform, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
