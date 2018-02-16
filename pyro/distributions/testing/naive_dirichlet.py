from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch import Beta, Dirichlet, Gamma
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Dirichlet)
class NaiveDirichlet(Dirichlet):
    """
    Implementation of ``Dirichlet`` via ``Gamma``.

    This naive implementation has stochastic reparameterized gradients, which
    have higher variance than PyTorch's ``Dirichlet`` implementation.
    """
    def __init__(self, alpha):
        super(NaiveDirichlet, self).__init__(alpha)
        self._gamma = Gamma(alpha, torch.ones_like(alpha))

    def rsample(self, sample_shape=torch.Size()):
        gammas = self._gamma.rsample(sample_shape)
        return gammas / gammas.sum(-1, True)


@copy_docs_from(Beta)
class NaiveBeta(Beta):
    """
    Implementation of ``Beta`` via ``Gamma``.

    This naive implementation has stochastic reparameterized gradients, which
    have higher variance than PyTorch's ``Beta`` implementation.
    """
    def __init__(self, alpha, beta):
        super(NaiveBeta, self).__init__(alpha, beta)
        alpha_beta = torch.stack([alpha, beta], -1)
        self._gamma = Gamma(alpha_beta, torch.ones_like(alpha_beta))

    def rsample(self, sample_shape=torch.Size()):
        gammas = self._gamma.rsample(sample_shape)
        probs = gammas / gammas.sum(-1, True)
        return probs[..., 0]
