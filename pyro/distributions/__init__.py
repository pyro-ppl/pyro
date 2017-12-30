from __future__ import absolute_import, division, print_function

import os

# Notice to Contributors. (@fritzo 2017-12-26)
#
# The Pyro team is moving pyro.distributions implementations upstream to
# torch.distributions, aiming for the PyTorch 0.4 release and Pyro 0.2 release
# in late Jan or Feb 2018.
# Tasks: https://github.com/probtorch/pytorch/projects/1
# Design Doc: https://goo.gl/9ccYsq
#
# To contribute new distributions you can either:
# 1. (Preferred) Implement a new distributions in torch.distributions and then
#    create a wrapper in pyro.distributions.torch.
# 2. Implement a new distribution in pyro.distribution and let Pyro devs move
#    this implementation upstream to torch.distributions.

# TODO move these implementations upstream to torch.distributions
from pyro.distributions.binomial import Binomial
from pyro.distributions.delta import Delta
from pyro.distributions.distribution import Distribution  # noqa: F401
from pyro.distributions.half_cauchy import HalfCauchy
from pyro.distributions.log_normal import LogNormal
from pyro.distributions.multinomial import Multinomial
from pyro.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions.poisson import Poisson
from pyro.distributions.random_primitive import RandomPrimitive

# distribution classes with working torch versions in torch.distributions
USE_TORCH_DISTRIBUTIONS = int(os.environ.get('PYRO_USE_TORCH_DISTRIBUTIONS', 0))
if USE_TORCH_DISTRIBUTIONS:
    from pyro.distributions.torch.bernoulli import Bernoulli
    from pyro.distributions.torch.beta import Beta
    from pyro.distributions.torch.categorical import Categorical
    from pyro.distributions.torch.cauchy import Cauchy
    from pyro.distributions.torch.dirichlet import Dirichlet
    from pyro.distributions.torch.exponential import Exponential
    from pyro.distributions.torch.gamma import Gamma
    from pyro.distributions.torch.normal import Normal
    from pyro.distributions.torch.one_hot_categorical import OneHotCategorical
    from pyro.distributions.torch.uniform import Uniform
else:
    from pyro.distributions.bernoulli import Bernoulli
    from pyro.distributions.beta import Beta
    from pyro.distributions.categorical import Categorical
    from pyro.distributions.cauchy import Cauchy
    from pyro.distributions.dirichlet import Dirichlet
    from pyro.distributions.exponential import Exponential
    from pyro.distributions.gamma import Gamma
    from pyro.distributions.normal import Normal
    from pyro.distributions.one_hot_categorical import OneHotCategorical
    from pyro.distributions.uniform import Uniform

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
halfcauchy = RandomPrimitive(HalfCauchy)
lognormal = RandomPrimitive(LogNormal)
multinomial = RandomPrimitive(Multinomial)
normal = RandomPrimitive(Normal)
one_hot_categorical = RandomPrimitive(OneHotCategorical)
poisson = RandomPrimitive(Poisson)
uniform = RandomPrimitive(Uniform)
multivariate_normal = RandomPrimitive(MultivariateNormal)
