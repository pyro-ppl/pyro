from __future__ import absolute_import, division, print_function

import os

from pyro.distributions.binomial import Binomial
from pyro.distributions.cauchy import Cauchy
from pyro.distributions.delta import Delta
from pyro.distributions.distribution import Distribution  # noqa: F401
from pyro.distributions.half_cauchy import HalfCauchy
from pyro.distributions.log_normal import LogNormal
from pyro.distributions.multinomial import Multinomial
from pyro.distributions.poisson import Poisson
from pyro.distributions.random_primitive import RandomPrimitive
from pyro.distributions.uniform import Uniform

# TODO Decide based on torch.__version__ once torch.distributions matures.
USE_TORCH_DISTRIBUTIONS = int(os.environ.get('PYRO_USE_TORCH_DISTRIBUTIONS', 0))

# distribution classes with working torch versions
if USE_TORCH_DISTRIBUTIONS:
    from pyro.distributions.torch.bernoulli import Bernoulli
    from pyro.distributions.torch.beta import Beta
    from pyro.distributions.torch.categorical import Categorical
    from pyro.distributions.torch.dirichlet import Dirichlet
    from pyro.distributions.torch.exponential import Exponential
    from pyro.distributions.torch.gamma import Gamma
    from pyro.distributions.torch.normal import Normal
    from pyro.distributions.torch.one_hot_categorical import OneHotCategorical
else:
    from pyro.distributions.bernoulli import Bernoulli
    from pyro.distributions.beta import Beta
    from pyro.distributions.categorical import Categorical
    from pyro.distributions.dirichlet import Dirichlet
    from pyro.distributions.exponential import Exponential
    from pyro.distributions.gamma import Gamma
    from pyro.distributions.normal import Normal
    from pyro.distributions.one_hot_categorical import OneHotCategorical

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
