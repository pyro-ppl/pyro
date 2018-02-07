from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Dirichlet(TorchDistribution):
    """
    Dirichlet distribution parameterized by a vector `alpha`.

    Dirichlet is a multivariate generalization of the Beta distribution.

    :param alpha:  *(real (0, Infinity))*
    """
    reparameterized = True

    def __init__(self, alpha, *args, **kwargs):
        """
        :param alpha: A vector of concentration parameters.
        :type alpha: None or a torch.autograd.Variable of a torch.Tensor of dimension 1 or 2.
        :param int batch_size: DEPRECATED.
        """
        torch_dist = torch.distributions.Dirichlet(alpha)
        super(Dirichlet, self).__init__(torch_dist, *args, **kwargs)

    def sample(self, sample_shape=torch.Size()):
        """
        Draws either a single sample (if alpha.dim() == 1), or one sample per param (if alpha.dim() == 2).

        (Un-reparameterized).

        :param torch.autograd.Variable alpha:
        """
        return super(Dirichlet, self).sample(sample_shape)

    def log_prob(self, x):
        """
        Evaluates log probability density over one or a batch of samples.

        Each of alpha and x can be either a single value or a batch of values batched along dimension 0.
        If they are both batches, their batch sizes must agree.
        In any case, the rightmost size must agree.

        :param torch.autograd.Variable x: A value (if x.dim() == 1) or or batch of values (if x.dim() == 2).
        :param alpha: A vector of concentration parameters.
        :type alpha: torch.autograd.Variable or None.
        :return: log probability densities of each element in the batch.
        :rtype: torch.autograd.Variable of torch.Tensor of dimension 1.
        """
        return super(Dirichlet, self).log_prob(x)
