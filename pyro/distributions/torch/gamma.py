from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(TorchDistribution)
class Gamma(TorchDistribution):
    """
    Gamma distribution parameterized by `alpha` and `beta`.

    This is often used in conjunction with `torch.nn.Softplus` to ensure
    `alpha` and `beta` parameters are positive.

    :param torch.autograd.Variable alpha: Shape parameter. Should be positive.
    :param torch.autograd.Variable beta: Shape parameter. Should be positive.
        Shouldb be the same shape as `alpha`.
    """
    reparameterized = True

    def __init__(self, alpha, beta, *args, **kwargs):
        torch_dist = torch.distributions.Gamma(alpha, beta)
        x_shape = torch.Size(broadcast_shape(alpha.size(), beta.size(), strict=True))
        event_dim = 1
        super(Gamma, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    @property
    def alpha(self):
        return self.torch_dist.concentration

    @property
    def beta(self):
        return self.torch_dist.rate
