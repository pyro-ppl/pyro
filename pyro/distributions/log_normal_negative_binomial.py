# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property

from pyro.distributions.torch import NegativeBinomial
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.ops.special import get_quad_rule


class LogNormalNegativeBinomial(TorchDistribution):
    r"""
    A three-parameter generalization of the Negative Binomial distribution [1].
    It can be understood as a continuous mixture of Negative Binomial distributions
    in which we inject Normally-distributed noise into the logits of the Negative
    Binomial distribution:

    .. math::

        \begin{eqnarray}
        &\rm{LNNB}(y | \rm{total\_count}=\nu, \rm{logits}=\ell, \rm{multiplicative\_noise\_scale}=sigma) = \\
        &\int d\epsilon \mathcal{N}(\epsilon | 0, \sigma)
        \rm{NB}(y | \rm{total\_count}=\nu, \rm{logits}=\ell + \epsilon)
        \end{eqnarray}

    where :math:`y \ge 0` is a non-negative integer. Thus while a Negative Binomial distribution
    can be formulated as a Poisson distribution with a Gamma-distributed rate, this distribution
    adds an additional level of variability by also modulating the rate by Log Normally-distributed
    multiplicative noise.

    This distribution has a mean given by

    .. math::
        \mathbb{E}[y] = \nu e^{\ell} = e^{\ell + \log \nu + \tfrac{1}{2}\sigma^2}

    and a variance given by

    .. math::
        \rm{Var}[y] = \mathbb{E}[y] + \left( e^{\sigma^2} (1 + 1/\nu) - 1 \right) \left( \mathbb{E}[y] \right)^2

    Thus while a given mean and variance together uniquely characterize a Negative Binomial distribution, there is a
    one-dimensional family of Log Normal Negative Binomial distributions with a given mean and variance.

    Note that in some applications it may be useful to parameterize the logits as

    .. math::
        \ell = \ell^\prime - \log \nu - \tfrac{1}{2}\sigma^2

    so that the mean is given by :math:`\mathbb{E}[y] = e^{\ell^\prime}` and does not depend on :math:`\nu`
    and :math:`\sigma`, which serve to determine the higher moments.

    References:

    [1] "Lognormal and Gamma Mixed Negative Binomial Regression,"
    Mingyuan Zhou, Lingbo Li, David Dunson, and Lawrence Carin.

    :param total_count: non-negative number of negative Bernoulli trials. The variance decreases
        as `total_count` increases.
    :type total_count: float or torch.Tensor
    :param torch.Tensor logits: Event log-odds for probabilities of success for underlying
        Negative Binomial distribution.
    :param torch.Tensor multiplicative_noise_scale: Controls the level of the injected Normal logit noise.
    :param int num_quad_points: Number of quadrature points used to compute the (approximate) `log_prob`.
        Defaults to 8.
    """

    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "logits": constraints.real,
        "multiplicative_noise_scale": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count,
        logits,
        multiplicative_noise_scale,
        *,
        num_quad_points=8,
        validate_args=None,
    ):
        if num_quad_points < 1:
            raise ValueError("num_quad_points must be positive.")

        total_count, logits, multiplicative_noise_scale = broadcast_all(
            total_count, logits, multiplicative_noise_scale
        )

        self.quad_points, self.log_weights = get_quad_rule(num_quad_points, logits)
        quad_logits = (
            logits.unsqueeze(-1)
            + multiplicative_noise_scale.unsqueeze(-1) * self.quad_points
        )
        self.nb_dist = NegativeBinomial(
            total_count=total_count.unsqueeze(-1), logits=quad_logits
        )

        self.multiplicative_noise_scale = multiplicative_noise_scale
        self.total_count = total_count
        self.logits = logits
        self.num_quad_points = num_quad_points

        batch_shape = broadcast_shape(
            multiplicative_noise_scale.shape, self.nb_dist.batch_shape[:-1]
        )
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        nb_log_prob = self.nb_dist.log_prob(value.unsqueeze(-1))
        return torch.logsumexp(self.log_weights + nb_log_prob, axis=-1)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        total_count = self.total_count.expand(batch_shape)
        logits = self.logits.expand(batch_shape)
        multiplicative_noise_scale = self.multiplicative_noise_scale.expand(batch_shape)
        LogNormalNegativeBinomial.__init__(
            new,
            total_count,
            logits,
            multiplicative_noise_scale,
            num_quad_points=self.num_quad_points,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def mean(self):
        return torch.exp(
            self.logits
            + self.total_count.log()
            + 0.5 * self.multiplicative_noise_scale.pow(2.0)
        )

    @lazy_property
    def variance(self):
        kappa = (
            torch.exp(self.multiplicative_noise_scale.pow(2.0))
            * (1 + 1 / self.total_count)
            - 1
        )
        return self.mean + kappa * self.mean.pow(2.0)
