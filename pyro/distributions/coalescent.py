# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.util import broadcast_shape
from pyro.ops.indexing import Vindex
from pyro.ops.tensor_utils import safe_log

from .torch import Exponential
from .torch_distribution import TorchDistribution


def _interpolate(array, x):
    x0 = x.floor()
    x1 = x0 + 1
    f0 = Vindex(array)[..., x0.long()]
    f1 = Vindex(array)[..., x1.long()]
    return f0 * (x1 - x) + f1 * (x - x0)


class CoalescentIntervals(TorchDistribution):
    """
    Distribution over coalescent time intervals in Kingman's ``n``-coalescent
    process [1,2].

    This samples over time intervals between successive coalescent events
    rather than the times, because the intervals are independent, making the
    constraint a simple ``constraints.positive``. Coalescent times can be
    computed as ``intervals.cumsum(-1)``::

        d = CoalescentIntervals(10)
        intervals = d.rsample()
        times = intervals.cumsum(-1)

    Note this has data-dependent ``event_shape = (num_leaves.max() - 1)``.
    When ``num_leaves`` is heterogeneous short intervals will be zero padded.

    **References**

    [1] J.F.C. Kingman (1982)
        "On the Genealogy of Large Populations"
        Journal of Applied Probability
    [2] J.F.C. Kingman (1982)
        "The Coalescent"
        Stochastic Processes and their Applications

    :param num_leaves: Number ``n`` of leaves (terminal nodes) in the final
        generation of coalescent process.
    :type num_leaves: int or torch.Tensor of float type
    """
    has_rsample = True
    arg_constraints = {"num_leaves": constraints.positive_integer}
    support = constraints.positive

    def __init__(self, num_leaves, *, validate_args=None):
        if isinstance(num_leaves, torch.Tensor):
            max_leaves = int(num_leaves.max())
        else:
            max_leaves = int(num_leaves)
            num_leaves = torch.tensor(float(num_leaves))
        self.num_leaves = num_leaves
        batch_shape = num_leaves.shape
        event_shape = (max_leaves - 1,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def rate(self):
        remaining = self.num_leaves.unsqueeze(-1) - torch.arange(self.event_shape[0])
        remaining = remaining.clamp(min=2)  # avoid nan during masking
        return remaining * (remaining - 1) * 0.5

    @lazy_property
    def event_mask(self):
        return self.num_leaves.unsqueeze(-1) - torch.arange(self.event_shape[0]) >= 2

    def expand(self, batch_shape, _instance=None):
        num_leaves = self.num_leaves.expand(batch_shape)
        new = self._get_checked_instance(CoalescentIntervals, _instance)
        CoalescentIntervals.__init__(new, num_leaves, validate_args=False)
        new._validate_args = self.__dict__.get("_validate_args")
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = Exponential(self.rate).log_prob(value)
        log_prob = log_prob.masked_fill(log_prob, ~self.event_mask, 0)
        return log_prob.sum(-1)

    def rsample(self, sample_shape):
        value = Exponential(self.rate).rsample(sample_shape)
        value = value.masked_fill(value, ~self.event_mask, 0)
        return value


class CoalescentTimes(TorchDistribution):
    """
    Distribution over coalescent times given irregular sampled leaves and
    piecewise constant coalescent rates defined on a regular time grid.

    **References**

    [1] J.F.C. Kingman (1982)
        "On the Genealogy of Large Populations"
        Journal of Applied Probability
    [2] J.F.C. Kingman (1982)
        "The Coalescent"
        Stochastic Processes and their Applications
    [3] A. Popinga, T. Vaughan, T. Statler, A.J. Drummond (2014)
        "Inferring epidemiological dynamics with Bayesian coalescentinference:
        The merits of deterministic and stochastic models"
        https://arxiv.org/pdf/1407.1792.pdf

    :param torch.Tensor leaves:
    :param torch.Tensor rates:
    """
    arg_constraints = {"leaves": constraints.real,
                       "rates": constraints.positive}

    def __init__(self, leaves, rates, *, validate_args=None):
        event_shape = (leaves.size(-1) - 1,)
        batch_shape = rates.shape[:-1]
        self.rates = rates
        self._unbroadcasted_rates = rates
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(0, self.rates.size(-1))

    @lazy_property
    def rates_cumsum(self):
        cumsum = self._unbroadcasted_rates.cumsum(-1)
        cumsum = torch.nn.functional.pad(cumsum, (1, 0), value=0)
        return cumsum

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CoalescentTimes, _instance)
        new.rates = self.rates.expand(batch_shape + (-1,))
        new._unbroadcasted_rates = self.rates
        super(CoalescentTimes, new).__init__(
            batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get("_validate_args")
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        finfo = torch.finfo(value.dtype)

        # Combine sampling events (self.leaves) and coalescent events (value)
        # into (times, signs) where samples are +1 and coalescences are -1.
        shape = broadcast_shape(value.shape[:-1], self.leaves.shape[:-1])
        leaves = self.leaves.expand(shape + (-1,))
        value = value.expand(shape + (-1,))
        times = torch.cat([value, leaves], dim=-1)
        signs = torch.arange(-value.size(-1), leaves.size(-1)).sign()  # e.g. [-1,1,1]
        times, idx = times.sort(dim=-1, descending=True)
        signs = signs.index_select(-1, idx)

        n = signs.cumsum(-1)
        rate = n * (n - 1) / 2

        # Compute survival factors for closed intervals.
        t = times.clamp(min=0)
        integrals = _interpolate(self.rates_cumsum, t)
        parts = integrals[..., 1:] - integrals[..., :-1]
        parts = parts.clamp(min=finfo.tiny)  # avoid nan
        log_prob = (rate * parts).sum(-1)  # FIXME

        # Compute survival factor for initial interval.
        # TODO

        # Compute density of coalescent events.
        rates = rate * _interpolate(self._unbroadcasted_rates, t)
        rates = rates.clamp(min=finfo.tiny)  # avoid nan
        log_prob = log_prob - signs.clamp(max=0) * safe_log(rates)

        return log_prob
