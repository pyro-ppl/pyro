# abstract base class
from pyro.distributions.distribution import Distribution

# specific distributions
from pyro.distributions.diag_normal import DiagNormal
from pyro.distributions.normal import Normal
from pyro.distributions.log_normal import LogNormal
from pyro.distributions.normal_chol import NormalChol
from pyro.distributions.uniform import Uniform
from pyro.distributions.poisson import Poisson
from pyro.distributions.gamma import Gamma
from pyro.distributions.beta import Beta
from pyro.distributions.bernoulli import Bernoulli
from pyro.distributions.multinomial import Multinomial
from pyro.distributions.exponential import Exponential
from pyro.distributions.categorical import Categorical
from pyro.distributions.delta import Delta
from pyro.distributions.transformed_distribution import TransformedDistribution
from pyro.distributions.transformed_distribution import AffineExp
from pyro.distributions.transformed_distribution import Bijector

# function aliases
diagnormal = DiagNormal()
lognormal = LogNormal()
categorical = Categorical()
bernoulli = Bernoulli()
beta = Beta()
delta = Delta()
exponential = Exponential()
gamma = Gamma()
multinomial = Multinomial()
normal = Normal()
normalchol = NormalChol()
poisson = Poisson()
uniform = Uniform()
