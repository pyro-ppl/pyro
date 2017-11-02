from __future__ import absolute_import, division, print_function

# distribution classes
from pyro.distributions.bernoulli import Bernoulli
from pyro.distributions.beta import Beta
from pyro.distributions.categorical import Categorical
from pyro.distributions.cauchy import Cauchy
from pyro.distributions.delta import Delta
from pyro.distributions.dirichlet import Dirichlet
from pyro.distributions.distribution import Distribution  # noqa: F401
from pyro.distributions.exponential import Exponential
from pyro.distributions.gamma import Gamma
from pyro.distributions.half_cauchy import HalfCauchy
from pyro.distributions.log_normal import LogNormal
from pyro.distributions.multinomial import Multinomial
from pyro.distributions.normal import Normal
from pyro.distributions.poisson import Poisson
from pyro.distributions.random_primitive import RandomPrimitive
from pyro.distributions.uniform import Uniform

# function aliases
bernoulli = RandomPrimitive(Bernoulli)
beta = RandomPrimitive(Beta)
categorical = RandomPrimitive(Categorical)
cauchy = RandomPrimitive(Cauchy)
delta = RandomPrimitive(Delta)
dirichlet = RandomPrimitive(Dirichlet)
exponential = RandomPrimitive(Exponential)
gamma = RandomPrimitive(Gamma)
halfcauchy = RandomPrimitive(HalfCauchy)
lognormal = RandomPrimitive(LogNormal)
multinomial = RandomPrimitive(Multinomial)
normal = RandomPrimitive(Normal)
poisson = RandomPrimitive(Poisson)
uniform = RandomPrimitive(Uniform)
