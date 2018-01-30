from __future__ import absolute_import, division, print_function

# Notice to Contributors. (@fritzo 2017-12-26)
#
# The Pyro team is moving pyro.distributions implementations upstream to
# torch.distributions, aiming for the PyTorch 0.4 release and Pyro 0.2 release
# around Feb 2018.
# Tasks: https://github.com/probtorch/pytorch/projects/1
# Design Doc: https://goo.gl/9ccYsq
#
# To contribute new distributions you can either:
# 1. (Preferred) Implement a new distributions in torch.distributions and then
#    create a wrapper in pyro.distributions.torch.
# 2. Implement a new distribution in pyro.distribution and let Pyro devs move
#    this implementation upstream to torch.distributions.

from pyro.distributions.delta import Delta
from pyro.distributions.distribution import Distribution  # noqa: F401
from pyro.distributions.random_primitive import RandomPrimitive
from pyro.distributions.rejector import Rejector  # noqa: F401

# distribution classes with working torch versions in torch.distributions
from pyro.distributions.torch.bernoulli import Bernoulli
from pyro.distributions.torch.beta import Beta
from pyro.distributions.torch.binomial import Binomial
from pyro.distributions.torch.categorical import Categorical
from pyro.distributions.torch.cauchy import Cauchy
from pyro.distributions.torch.dirichlet import Dirichlet
from pyro.distributions.torch.exponential import Exponential
from pyro.distributions.torch.gamma import Gamma
from pyro.distributions.torch.log_normal import LogNormal
from pyro.distributions.torch.multinomial import Multinomial
from pyro.distributions.torch.multivariate_normal import MultivariateNormal
from pyro.distributions.torch.normal import Normal
from pyro.distributions.torch.one_hot_categorical import OneHotCategorical
from pyro.distributions.torch.poisson import Poisson
from pyro.distributions.torch.uniform import Uniform

# function aliases
bernoulli = RandomPrimitive(Bernoulli)
beta = RandomPrimitive(Beta)
binomial = RandomPrimitive(Binomial)
categorical = RandomPrimitive(Categorical)
cauchy = RandomPrimitive(Cauchy)
delta = RandomPrimitive(Delta)
dirichlet = RandomPrimitive(Dirichlet)
exponential = RandomPrimitive(Exponential)
gamma = RandomPrimitive(Gamma)
lognormal = RandomPrimitive(LogNormal)
multinomial = RandomPrimitive(Multinomial)
normal = RandomPrimitive(Normal)
one_hot_categorical = RandomPrimitive(OneHotCategorical)
poisson = RandomPrimitive(Poisson)
uniform = RandomPrimitive(Uniform)
multivariate_normal = RandomPrimitive(MultivariateNormal)
