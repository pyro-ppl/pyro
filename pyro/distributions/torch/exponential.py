from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Exponential(TorchDistribution):
    """
    Exponential parameterized by scale `lambda`.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `lam` parameter is positive.

    :param torch.autograd.Variable lam: Scale parameter (a.k.a. `lambda`).
        Should be positive.
    """
    reparameterized = True

    def __init__(self, lam, *args, **kwargs):
        torch_dist = torch.distributions.Exponential(lam)
        super(Exponential, self).__init__(torch_dist, *args, **kwargs)
