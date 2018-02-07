from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Beta(TorchDistribution):
    """
    Univariate beta distribution parameterized by `alpha` and `beta`.

    This is often used in conjunction with `torch.nn.Softplus` to ensure
    `alpha` and `beta` parameters are positive.

    :param torch.autograd.Variable alpha: Lower shape parameter.
        Should be positive.
    :param torch.autograd.Variable beta: Upper shape parameter.
        Should be positive.
    """
    reparameterized = True

    def __init__(self, alpha, beta, *args, **kwargs):
        torch_dist = torch.distributions.Beta(alpha, beta)
        super(Beta, self).__init__(torch_dist, *args, **kwargs)
