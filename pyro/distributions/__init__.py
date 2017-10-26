# abstract base class
from pyro.distributions.bernoulli import Bernoulli
from pyro.distributions.beta import Beta
from pyro.distributions.categorical import Categorical
from pyro.distributions.cauchy import Cauchy
from pyro.distributions.half_cauchy import HalfCauchy
from pyro.distributions.delta import Delta
from pyro.distributions.distribution import Distribution  # noqa: F401
# specific distributions
from pyro.distributions.diag_normal import DiagNormal
from pyro.distributions.dirichlet import Dirichlet
from pyro.distributions.exponential import Exponential
from pyro.distributions.gamma import Gamma
from pyro.distributions.log_normal import LogNormal
from pyro.distributions.multinomial import Multinomial
from pyro.distributions.poisson import Poisson
from pyro.distributions.random_primitive import RandomPrimitive
from pyro.distributions.uniform import Uniform

# function aliases
diagnormal = DiagNormal()
lognormal = RandomPrimitive(LogNormal)
categorical = Categorical()
bernoulli = RandomPrimitive(Bernoulli)
beta = RandomPrimitive(Beta)
delta = RandomPrimitive(Delta)
exponential = RandomPrimitive(Exponential)
gamma = RandomPrimitive(Gamma)
multinomial = RandomPrimitive(Multinomial)
poisson = RandomPrimitive(Poisson)
uniform = RandomPrimitive(Uniform)
dirichlet = RandomPrimitive(Dirichlet)
cauchy = RandomPrimitive(Cauchy)
halfcauchy = RandomPrimitive(HalfCauchy)
