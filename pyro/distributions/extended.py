# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

from pyro.distributions import constraints

from .conjugate import BetaBinomial
from .torch import Binomial


class ExtendedBinomial(Binomial):
    """
    EXPERIMENTAL :class:`~pyro.distributions.Binomial` distribution extended to
    have logical support the entire integers. Numerical support is still the
    integer interval ``[0, total_count]``
    """
    support = constraints.integer

    def log_prob(self, value):
        result = super().log_prob(value)
        invalid = ~super().support.check(value)
        return result.masked_fill(invalid, -math.inf)


class ExtendedBetaBinomial(BetaBinomial):
    """
    EXPERIMENTAL :class:`~pyro.distributions.BetaBinomial` distribution
    extended to have logical support the entire integers. Numerical support is
    still the integer interval ``[0, total_count]``
    """
    support = constraints.integer

    def log_prob(self, value):
        result = super().log_prob(value)
        invalid = ~super().support.check(value)
        return result.masked_fill(invalid, -math.inf)
