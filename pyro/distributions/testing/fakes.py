from __future__ import absolute_import, division, print_function

from pyro.distributions.random_primitive import RandomPrimitive
from pyro.distributions.torch.beta import Beta
from pyro.distributions.torch.dirichlet import Dirichlet
from pyro.distributions.torch.gamma import Gamma
from pyro.distributions.torch.normal import Normal


class NonreparameterizedBeta(Beta):
    reparameterized = False


class NonreparameterizedDirichlet(Dirichlet):
    reparameterized = False


class NonreparameterizedGamma(Gamma):
    reparameterized = False


class NonreparameterizedNormal(Normal):
    reparameterized = False


nonreparameterized_beta = RandomPrimitive(NonreparameterizedBeta)
nonreparameterized_dirichlet = RandomPrimitive(NonreparameterizedDirichlet)
nonreparameterized_gamma = RandomPrimitive(NonreparameterizedGamma)
nonreparameterized_normal = RandomPrimitive(NonreparameterizedNormal)
