import numpy as np
from numpy.polynomial.hermite import hermgauss

import torch
from torch.distributions import constraints

from pyro.distributions import NegativeBinomial, TorchDistribution
from pyro.distributions.util import broadcast_shape


def get_quad_rule(num_quad, prototype_tensor):
    quad_rule = hermgauss(num_quad)
    quad_points = quad_rule[0] * np.sqrt(2.0)
    log_weights = np.log(quad_rule[1]) - 0.5 * np.log(np.pi)
    return torch.from_numpy(quad_points).type_as(prototype_tensor), \
        torch.from_numpy(log_weights).type_as(prototype_tensor)


class LogNormalNegativeBinomial(TorchDistribution):
    """
    A three-parameter generalization of the Negative Binomial distribution [1].
    It can be understood as a continuous mixture of Negative Binomial distributions
    in which we inject Normally-distributed noise into the logits of the Negative
    Binomial distribution:

    :math:`\rm{LNNB}(\rm{total_count}=\nu, \rm{logits}=\ell, \rm{multiplicative_noise_scale}=sigma) = \int d\epsilon
    \mathcal{N}(\epsilon | 0, \sigma) \rm{NB}(\rm{total_count}=\nu, \rm{logits}=\ell + \epsilon)`

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
