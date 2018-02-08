from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Cauchy(TorchDistribution):
    """
    Cauchy (a.k.a. Lorentz) distribution.

    This is a continuous distribution which is roughly the ratio of two
    Gaussians if the second Gaussian is zero mean. The distribution is over
    tensors that have the same shape as the parameters `mu`and `gamma`, which
    in turn must have the same shape as each other.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `gamma` parameter is positive.

    :param torch.autograd.Variable mu: Location parameter.
    :param torch.autograd.Variable gamma: Scale parameter. Should be positive.
    """
    reparameterized = True

    def __init__(self, mu, gamma, *args, **kwargs):
        torch_dist = torch.distributions.Cauchy(mu, gamma)
        super(Cauchy, self).__init__(torch_dist, *args, **kwargs)
