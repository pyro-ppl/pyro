from __future__ import absolute_import, division, print_function

from pyro.distributions.torch import Beta, Dirichlet, Gamma, Normal


class NonreparameterizedBeta(Beta):
    has_rsample = False


class NonreparameterizedDirichlet(Dirichlet):
    has_rsample = False


class NonreparameterizedGamma(Gamma):
    has_rsample = False


class NonreparameterizedNormal(Normal):
    has_rsample = False
