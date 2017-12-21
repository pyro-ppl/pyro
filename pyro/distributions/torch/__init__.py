from __future__ import absolute_import, division, print_function

from pyro.distributions.distribution import Distribution  # noqa: F401
from pyro.distributions.random_primitive import RandomPrimitive
from pyro.distributions.torch.beta import Beta
from pyro.distributions.torch.categorical import Categorical
from pyro.distributions.torch.dirichlet import Dirichlet
from pyro.distributions.torch.exponential import Exponential
from pyro.distributions.torch.gamma import Gamma
from pyro.distributions.torch.normal import Normal

# function aliases
beta = RandomPrimitive(Beta)
categorical = RandomPrimitive(Categorical)
dirichlet = RandomPrimitive(Dirichlet)
exponential = RandomPrimitive(Exponential)
gamma = RandomPrimitive(Gamma)
normal = RandomPrimitive(Normal)
