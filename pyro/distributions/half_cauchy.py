from __future__ import absolute_import, division, print_function

import math

from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform, AffineTransform
from torch.distributions.utils import broadcast_all

from pyro.distributions.torch import Cauchy, TransformedDistribution


class HalfCauchy(TransformedDistribution):
    r"""
    Half-Cauchy distribution.

    This is a continuous distribution with lower-bounded domain (`x > loc`).
    See also the :class:`~pyro.distributions.torch.Cauchy` distribution.

    :param torch.Tensor loc: lower bound of the distribution.
    :param torch.Tensor scale: half width at half maximum.
    """
    arg_constraints = Cauchy.arg_constraints
    support = Cauchy.support

    def __init__(self, loc, scale):
        loc, scale = broadcast_all(loc, scale)
        base_dist = Cauchy(0, scale)
        transforms = [AbsTransform(), AffineTransform(loc, 1)]
        super(HalfCauchy, self).__init__(base_dist, transforms)

    @property
    def loc(self):
        return self.transforms[1].loc

    @property
    def scale(self):
        return self.base_dist.scale

    @constraints.dependent_property
    def support(self):
        return constraints.greater_than(self.loc)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value - self.loc) + math.log(2)
        log_prob[value < self.loc] = -float('inf')
        return log_prob

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)

    def expand(self, batch_shape):
        loc = self.loc.expand(batch_shape)
        scale = self.scale.expand(batch_shape)
        return HalfCauchy(loc, scale)
