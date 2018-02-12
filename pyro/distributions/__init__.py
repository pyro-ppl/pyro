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

from __future__ import absolute_import, division, print_function

from pyro.distributions.delta import Delta
from pyro.distributions.distribution import Distribution
from pyro.distributions.rejector import Rejector
from pyro.distributions.torch.bernoulli import Bernoulli
from pyro.distributions.torch.beta import Beta
from pyro.distributions.torch.binomial import Binomial
from pyro.distributions.torch.categorical import Categorical
from pyro.distributions.torch.cauchy import Cauchy
from pyro.distributions.torch.dirichlet import Dirichlet
from pyro.distributions.torch.exponential import Exponential
from pyro.distributions.torch.gamma import Gamma
from pyro.distributions.torch.iaf import InverseAutoregressiveFlow
from pyro.distributions.torch.log_normal import LogNormal
from pyro.distributions.torch.multinomial import Multinomial
from pyro.distributions.torch.multivariate_normal import MultivariateNormal
from pyro.distributions.torch.normal import Normal
from pyro.distributions.torch.one_hot_categorical import OneHotCategorical
from pyro.distributions.torch.poisson import Poisson
from pyro.distributions.torch.sparse_multivariate_normal import SparseMultivariateNormal
from pyro.distributions.torch.transformed_distribution import TransformedDistribution
from pyro.distributions.torch.uniform import Uniform

# flake8: noqa
