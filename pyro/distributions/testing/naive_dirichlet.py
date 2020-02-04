# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

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
    def __init__(self, concentration, validate_args=None):
        super().__init__(concentration)
        self._gamma = Gamma(concentration, torch.ones_like(concentration), validate_args=validate_args)

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
    def __init__(self, concentration1, concentration0, validate_args=None):
        super().__init__(concentration1, concentration0, validate_args=validate_args)
        alpha_beta = torch.stack([concentration1, concentration0], -1)
        self._gamma = Gamma(alpha_beta, torch.ones_like(alpha_beta))

    def rsample(self, sample_shape=torch.Size()):
        gammas = self._gamma.rsample(sample_shape)
        probs = gammas / gammas.sum(-1, True)
        return probs[..., 0]
