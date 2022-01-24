import numpy as np
from numpy.polynomial.hermite import hermgauss

import torch
from torch.distributions import constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.torch import NegativeBinomial
from pyro.distributions.util import broadcast_shape


def get_quad_rule(num_quad, prototype_tensor):
    quad_rule = hermgauss(num_quad)
    quad_points = quad_rule[0] * np.sqrt(2.0)
    log_weights = np.log(quad_rule[1]) - 0.5 * np.log(np.pi)
    return torch.from_numpy(quad_points).type_as(prototype_tensor), \
        torch.from_numpy(log_weights).type_as(prototype_tensor)


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

    where :math:`y \ge 0` is a non-negative integer. This distribution has a mean given by

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

    :param total_count: non-negative number of negative Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param torch.Tensor logits: Event log-odds for probabilities of success for underlying
        Negative Binomial distribution.
    :param num_quad_points: Number of quadrature points used to compute the (approximate) `log_prob`.
        Defaults to 8.
    :type num_quad_points: int
    """
    arg_constraints = {'total_count': constraints.greater_than_eq(0),
                       'logits': constraints.real,
                       'multiplicative_noise_scale': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, total_count, logits, multiplicative_noise_scale,
                 num_quad_points=8, validate_args=None):
        self.quad_points, self.log_weights = get_quad_rule(num_quad_points, logits)
        quad_logits = logits.unsqueeze(-1) + multiplicative_noise_scale.unsqueeze(-1) * self.quad_points
        self.nb = NegativeBinomial(total_count=total_count.unsqueeze(-1), logits=quad_logits)
        self.multiplicative_noise_scale = multiplicative_noise_scale

    def log_prob(self, value):
        nb_log_prob = self.nb.log_prob(value.unsqueeze(-1))
        return torch.logsumexp(self.log_weights + nb_log_prob, axis=-1)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError
