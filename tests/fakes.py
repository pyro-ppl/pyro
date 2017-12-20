from __future__ import absolute_import, division, print_function

from pyro.distributions.normal import Normal
from pyro.distributions.random_primitive import RandomPrimitive


class NonreparameterizedNormal(Normal):
    reparameterized = False


nonreparameterized_normal = RandomPrimitive(NonreparameterizedNormal)
