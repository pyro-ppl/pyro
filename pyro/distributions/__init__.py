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
from pyro.distributions.reshape import Reshape
from pyro.distributions.torch import (Bernoulli, Beta, Binomial, Categorical, Cauchy, Dirichlet, Exponential, Gamma,
                                      LogNormal, Multinomial, Normal, OneHotCategorical, Poisson,
                                      TransformedDistribution, Uniform)
from pyro.distributions.torch.iaf import InverseAutoregressiveFlow
from pyro.distributions.torch.multivariate_normal import MultivariateNormal
from pyro.distributions.torch.sparse_multivariate_normal import SparseMultivariateNormal

# flake8: noqa
