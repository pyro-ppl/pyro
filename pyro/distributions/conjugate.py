from __future__ import absolute_import, division, print_function

import numbers

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions.torch import Beta, Binomial, Dirichlet, Gamma, Multinomial, Poisson
from pyro.distributions.torch_distribution import TorchDistribution


def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


def _log_beta_1(alpha, value, is_sparse):
    if is_sparse:
        mask = (value != 0)
        value, alpha, mask = torch.broadcast_tensors(value, alpha, mask)
        result = torch.zeros_like(value)
        value = value[mask]
        alpha = alpha[mask]
        result[mask] = torch.lgamma(1 + value) + torch.lgamma(alpha) - torch.lgamma(value + alpha)
        return result
    else:
        return torch.lgamma(1 + value) + torch.lgamma(alpha) - torch.lgamma(value + alpha)


class BetaBinomial(TorchDistribution):
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
    support = Binomial.support

    def __init__(self, concentration1, concentration0, total_count=1, validate_args=None):
        concentration1, concentration0, total_count = broadcast_all(
            concentration1, concentration0, total_count)
        self._beta = Beta(concentration1, concentration0)
        self.total_count = total_count
        super(BetaBinomial, self).__init__(self._beta._batch_shape, validate_args=validate_args)

    @property
    def concentration1(self):
        return self._beta.concentration1

    @property
    def concentration0(self):
        return self._beta.concentration0

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BetaBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new._beta = self._beta.expand(batch_shape)
        new.total_count = self.total_count.expand_as(new._beta.concentration0)
        super(BetaBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        probs = self._beta.sample(sample_shape)
        return Binomial(self.total_count, probs).sample()

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
        return self._beta.mean * self.total_count

    @property
    def variance(self):
        return self._beta.variance * self.total_count * (self.concentration0 + self.concentration1 + self.total_count)

    def enumerate_support(self, expand=True):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `enumerate_support`.")
        values = torch.arange(1 + total_count, dtype=self.concentration1.dtype, device=self.concentration1.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values


class DirichletMultinomial(TorchDistribution):
    r"""
    Compound distribution comprising of a dirichlet-multinomial pair. The probability of
    classes (``probs`` for the :class:`~pyro.distributions.Multinomial` distribution)
    is unknown and randomly drawn from a :class:`~pyro.distributions.Dirichlet`
    distribution prior to a certain number of Categorical trials given by
    ``total_count``.

    :param float or torch.Tensor concentration: concentration parameter (alpha) for the
        Dirichlet distribution.
    :param int or torch.Tensor total_count: number of Categorical trials.
    :param bool is_sparse: Whether to assume value is mostly zero when computing
        :meth:`log_prob`, which can speed up computation when data is sparse.
    """
    arg_constraints = {'concentration': constraints.positive, 'total_count': constraints.nonnegative_integer}
    support = Multinomial.support

    def __init__(self, concentration, total_count=1, is_sparse=False, validate_args=None):
        if isinstance(total_count, numbers.Number):
            total_count = torch.tensor(total_count, dtype=concentration.dtype, device=concentration.device)
        total_count_1 = total_count.unsqueeze(-1)
        concentration, total_count = torch.broadcast_tensors(concentration, total_count_1)
        total_count = total_count_1.squeeze(-1)
        self._dirichlet = Dirichlet(concentration)
        self.total_count = total_count
        self.is_sparse = is_sparse
        super(DirichletMultinomial, self).__init__(
            self._dirichlet._batch_shape, self._dirichlet.event_shape, validate_args=validate_args)

    @property
    def concentration(self):
        return self._dirichlet.concentration

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DirichletMultinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.is_sparse = self.is_sparse
        super(DirichletMultinomial, new).__init__(
            new._dirichlet.batch_shape, new._dirichlet.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        probs = self._dirichlet.sample(sample_shape)
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `sample`.")
        return Multinomial(total_count, probs).sample()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        alpha = self.concentration
        return (_log_beta_1(alpha.sum(-1), value.sum(-1), self.is_sparse) -
                _log_beta_1(alpha, value, self.is_sparse).sum(-1))

    @property
    def mean(self):
        return self._dirichlet.mean * self.total_count.unsqueeze(-1)

    @property
    def variance(self):
        n = self.total_count.unsqueeze(-1)
        alpha = self.concentration
        alpha_sum = self.concentration.sum(-1, keepdim=True)
        alpha_ratio = alpha / alpha_sum
        return n * alpha_ratio * (1 - alpha_ratio) * (n + alpha_sum) / (1 + alpha_sum)


class GammaPoisson(TorchDistribution):
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
    support = Poisson.support

    def __init__(self, concentration, rate, validate_args=None):
        concentration, rate = broadcast_all(concentration, rate)
        self._gamma = Gamma(concentration, rate)
        super(GammaPoisson, self).__init__(self._gamma._batch_shape, validate_args=validate_args)

    @property
    def concentration(self):
        return self._gamma.concentration

    @property
    def rate(self):
        return self._gamma.rate

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GammaPoisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new._gamma = self._gamma.expand(batch_shape)
        super(GammaPoisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        rate = self._gamma.sample(sample_shape)
        return Poisson(rate).sample()

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
