# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

from pyro.distributions import constraints

from .conjugate import BetaBinomial
from .torch import Binomial


class ExtendedBinomial(Binomial):
    """
    EXPERIMENTAL :class:`~pyro.distributions.Binomial` distribution extended to
    have logical support the entire integers and to allow arbitrary integer
    ``total_count``. Numerical support is still the integer interval ``[0,
    total_count]``.
    """
    arg_constraints = {"total_count": constraints.integer,
                       "probs": constraints.unit_interval,
                       "logits": constraints.real}
    support = constraints.integer

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        invalid = (value < 0) | (value > self.total_count)
        result = super().log_prob(value)
        return result.masked_fill(invalid, -math.inf)


class ExtendedBetaBinomial(BetaBinomial):
    """
    EXPERIMENTAL :class:`~pyro.distributions.BetaBinomial` distribution
    extended to have logical support the entire integers and to allow arbitrary
    integer ``total_count``. Numerical support is still the integer interval
    ``[0, total_count]``.
    """
    arg_constraints = {"concentration1": constraints.positive,
                       "concentration0": constraints.positive,
                       "total_count": constraints.integer}
    support = constraints.integer

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        invalid = (value < 0) | (value > self.total_count)
        result = super().log_prob(value)
        return result.masked_fill(invalid, -math.inf)
