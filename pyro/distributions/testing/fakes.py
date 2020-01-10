# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.distributions.torch import Beta, Dirichlet, Gamma, Normal


class NonreparameterizedBeta(Beta):
    has_rsample = False


class NonreparameterizedDirichlet(Dirichlet):
    has_rsample = False


class NonreparameterizedGamma(Gamma):
    has_rsample = False


class NonreparameterizedNormal(Normal):
    has_rsample = False
