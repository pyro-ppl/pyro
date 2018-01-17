from __future__ import absolute_import, division, print_function

from pyro.distributions import Beta
from pyro.distributions import Gamma
from pyro.distributions import Normal
from pyro.distributions.random_primitive import RandomPrimitive


class NonreparameterizedBeta(Beta):
    reparameterized = False


class NonreparameterizedGamma(Gamma):
    reparameterized = False


class NonreparameterizedNormal(Normal):
    reparameterized = False


nonreparameterized_beta = RandomPrimitive(NonreparameterizedBeta)
nonreparameterized_gamma = RandomPrimitive(NonreparameterizedGamma)
nonreparameterized_normal = RandomPrimitive(NonreparameterizedNormal)
