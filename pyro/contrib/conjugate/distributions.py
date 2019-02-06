from __future__ import absolute_import, division, print_function

from operator import mul

from six.moves import reduce
import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

import pyro
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import sum_leftmost


def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


class ConjugateDistribution(TorchDistribution):
    def __init__(self, name, latent_dist, batch_shape=torch.Size(), event_shape=torch.Size(), validate_args=None):
        self.name = name
        self.latent = latent_dist
        self._latent_sample = None
        super(ConjugateDistribution, self).__init__(batch_shape, event_shape,
                                                    validate_args=validate_args)

    def pin_latent(self):
        self._latent_sample = pyro.sample(self.name + ".latent", self.latent,
                                          infer={"collapsed": True})
        self.latent = self.latent.expand(self._latent_sample.shape)

    def observe(self, value):
        return pyro.sample(self.name, self, obs=value)


class BetaBinomial(ConjugateDistribution):
    r"""
    Compound distribution comprising of a beta-binomial pair. The probability of
    success (``probs`` for the :class:`~pyro.distributions.Binomial` distribution)
    is unknown and randomly drawn from a :class:`~pyro.distributions.Beta` distribution
    prior to a certain number of Bernoulli trials given by ``total_count``.

    :param float or torch.Tensor concentration1: 1st concentration parameter (alpha) for the
        Beta distribution.
    :param float or torch.Tensor concentration0: 2nd concentration parameter (beta) for the
        Beta distribution.
    :param int or torch.Tensor total_count: number of Bernoulli trials.
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive,
                       'total_count': constraints.nonnegative_integer}
    has_enumerate_support = True
    support = dist.Binomial.support

    def __init__(self, name, concentration1, concentration0, total_count=1, validate_args=None):
        concentration1, concentration0, total_count = broadcast_all(
            concentration1, concentration0, total_count)
        beta = dist.Beta(concentration1, concentration0)
        self.total_count = total_count
        super(BetaBinomial, self).__init__(name, beta, beta._batch_shape, validate_args=validate_args)

    @property
    def concentration1(self):
        return self.latent.concentration1

    @property
    def concentration0(self):
        return self.latent.concentration0

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BetaBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.latent = self.latent
        new.total_count = self.total_count.expand(batch_shape)
        new.name = self.name
        super(BetaBinomial, new).__init__(new.name, new.latent, batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=(), internal_dispatch=False):
        if not internal_dispatch:
            raise RuntimeError("")
        probs = self.latent.sample(sample_shape)
        return dist.Binomial(self.total_count, probs).sample()

    def _posterior_latent(self, obs):
        reduce_dims = len(obs.size()) - len(self.concentration1.size())
        num_obs = reduce(mul, obs.size()[:reduce_dims], 1)
        total_count = self.total_count[tuple(slice(1) if i < reduce_dims else slice(None)
                                       for i in range(self.total_count.dim()))].reshape(self.concentration0.shape)
        summed_obs = sum_leftmost(obs, reduce_dims)
        return dist.Beta(self.concentration1 + summed_obs,
                         num_obs * total_count + self.concentration0 - summed_obs,
                         validate_args=self._validate_args)

    def _posterior_predictive(self, probs):
        return dist.Binomial(total_count=self.total_count, probs=probs, validate_args=self._validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                _log_beta(value + self.concentration1, self.total_count - value + self.concentration0) -
                _log_beta(self.concentration0, self.concentration1))

    @property
    def mean(self):
        return self.latent.mean * self.total_count

    @property
    def variance(self):
        return self.latent.variance * self.total_count * (self.concentration0 + self.concentration1 + self.total_count)

    def enumerate_support(self, expand=True):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `enumerate_support`.")
        values = torch.arange(1 + total_count, dtype=self.concentration1.dtype, device=self.concentration1.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values


class GammaPoisson(ConjugateDistribution):
    r"""
    Compound distribution comprising of a gamma-poisson pair, also referred to as
    a gamma-poisson mixture. The ``rate`` parameter for the
    :class:`~pyro.distributions.Poisson` distribution is unknown and randomly
    drawn from a :class:`~pyro.distributions.Gamma` distribution.

    .. note:: This can be treated as an alternate parametrization of the
        :class:`~pyro.distributions.NegativeBinomial` (``total_count``, ``probs``)
        distribution, with `concentration = total_count` and `rate = (1 - probs) / probs`.

    :param float or torch.Tensor concentration: shape parameter (alpha) of the Gamma
        distribution.
    :param float or torch.Tensor rate: rate parameter (beta) for the Gamma
        distribution.
    """

    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = dist.Poisson.support

    def __init__(self, name, concentration, rate, validate_args=None):
        concentration, rate = broadcast_all(concentration, rate)
        gamma = dist.Gamma(concentration, rate)
        super(GammaPoisson, self).__init__(name, gamma, gamma._batch_shape, validate_args=validate_args)

    @property
    def concentration(self):
        return self.latent.concentration

    @property
    def rate(self):
        return self.latent.rate

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GammaPoisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.latent = self.latent
        new.name = self.name
        super(GammaPoisson, new).__init__(new.name, new.latent, batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _posterior_latent(self, obs):
        reduce_dims = len(obs.size()) - len(self.rate.size())
        num_obs = reduce(mul, obs.size()[:reduce_dims], 1)
        summed_obs = sum_leftmost(obs, reduce_dims)
        return dist.Gamma(self.concentration + summed_obs, self.rate + num_obs)

    def _posterior_predictive(self, rate):
        return dist.Poisson(rate=rate, validate_args=self._validate_args)

    def sample(self, sample_shape=()):
        rate = self.latent.sample(sample_shape)
        return dist.Poisson(rate).sample()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        post_value = self.concentration + value
        return -_log_beta(self.concentration, value + 1) - post_value.log() + \
            self.concentration * self.rate.log() - post_value * (1 + self.rate).log()

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / self.rate.pow(2) * (1 + self.rate)
