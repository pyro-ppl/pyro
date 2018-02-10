from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Dirichlet(TorchDistribution):
    """
    Dirichlet distribution parameterized by a vector `alpha`.

    Dirichlet is a multivariate generalization of the Beta distribution.

    :param alpha: A vector of concentration parameters. Should be positive.
    :type alpha: None or a torch.autograd.Variable of a torch.Tensor of dimension 1 or 2.
    """
    reparameterized = True

    def __init__(self, alpha, *args, **kwargs):
        torch_dist = torch.distributions.Dirichlet(alpha)
        super(Dirichlet, self).__init__(torch_dist, *args, **kwargs)
