import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution


def _unsafe_standard_stable(shape, alpha, beta):
    # Implements a noisily reparametrized version of the sampler
    # Chambers-Mallows-Stuck method as corrected by Weron [1,3] and simplified
    # by Nolan [4]. This will fail if alpha is close to 1.

    # Draw parameter-free noise.
    with torch.no_grad():
        half_pi = math.pi / 2
        V = alpha.new_empty(shape).uniform_(-half_pi, half_pi)
        W = alpha.new_empty(shape).exponential_()

    # Differentiably transform noise via parameters.
    inv_alpha = alpha.reciprocal()
    b = beta * (half_pi * alpha).tan()
    v = b.atan() + alpha * V
    Z = v.sin() / ((1 + b * b).rsqrt() * V.cos()).pow(inv_alpha) \
        * ((v - V).cos() / W).pow(inv_alpha - 1)

    # Convert to Nolan's parametrization S^0 so that samples depend
    # continuously on (alpha,beta), allowing us to interpolate around the hole
    # at alpha=1.
    X = Z - b
    return X


RADIUS = 0.01


def _standard_stable(shape, alpha, beta):
    with torch.no_grad():
        hole = 1.
        near_hole = (alpha - hole).abs() <= RADIUS
    if not torch._C._get_tracing_state() and not near_hole.any():
        return _unsafe_standard_stable(shape, alpha, beta)

    # Avoid the hole at alpha=1 by interpolating between pairs
    # of points at hole-RADIUS and hole+RADIUS.
    shape_ = shape + (1,)
    beta_ = beta.unsqueeze(-1)
    alpha_ = alpha.unsqueeze(-1).expand(alpha.shape + (2,)).contiguous()
    with torch.no_grad():
        lower, upper = alpha_.unbind(-1)
        lower.data[near_hole] = hole - RADIUS
        upper.data[near_hole] = hole + RADIUS
        # We don't need to backprop through weights, since we've pretended
        # alpha_ is reparametrized, even though we've clamped some values.
        #               |a - a'|
        # weight = 1 - ----------
        #              2 * RADIUS
        weights = (alpha_ - alpha.unsqueeze(-1)).abs_().mul_(-1 / (2 * RADIUS)).add_(1)
        weights[~near_hole] = 0.5
    pairs = _unsafe_standard_stable(shape_, alpha_, beta_)
    return (pairs * weights).sum(-1)


class Stable(TorchDistribution):
    r"""
    Levy :math:`\alpha`-stable distribution. See [1] for a review.

    This uses Nolan's parametrization [2] of the ``loc`` parameter, which is
    required for continuity and differentiability. This corresponds to the
    notation :math:`S^0_\alpha(\beta,\sigma,\mu_0)` of [1], where
    :math:`\alpha` = stability, :math:`\beta` = skew, :math:`\sigma` = scale,
    and :math:`\mu_0` = loc.

    This implements a reparametrized sampler :meth:`rsample` , but does not
    implement :meth:`log_prob` . Use in inference is thus limited to
    likelihood-free algorithms such as
    :class:`~pyro.infer.energy_distance.EnergyDistance`.

    [1] S. Borak, W. Hardle, R. Weron (2005).
        Stable distributions.
        https://edoc.hu-berlin.de/bitstream/handle/18452/4526/8.pdf
    [2] J.P. Nolan (1997).
        Numerical calculation of stable densities and distribution functions.
    [3] Rafal Weron (1996).
        On the Chambers-Mallows-Stuck Method for
        Simulating Skewed Stable Random Variables.
    [4] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        http://fs2.american.edu/jpnolan/www/stable/chap1.pdf

    :param Tensor stability: Levy stability parameter :math:`\alpha\in(0,2]` .
    :param Tensor skew: Skewness :math:`\beta\in[-1,1]` .
    :param Tensor scale: Scale :math:`\sigma > 0` . Defaults to 1.
    :param Tensor loc: Location :math:`\mu_0` in Nolan's parametrization [2].
        Defaults to 0.
    """
    has_rsample = True
    arg_constraints = {"stability": constraints.interval(0, 2),  # half-open (0, 2]
                       "skew": constraints.interval(-1, 1),  # closed [-1, 1]
                       "scale": constraints.positive,
                       "loc": constraints.real}
    support = constraints.real

    def __init__(self, stability, skew, scale=1.0, loc=0.0, validate_args=None):
        self.stability, self.skew, self.scale, self.loc = broadcast_all(
            stability, skew, scale, loc)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Stable, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        raise NotImplementedError("Stable.log_prob() is not implemented")

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        x = _standard_stable(shape, self.stability, self.skew)
        return self.loc + self.scale * x
