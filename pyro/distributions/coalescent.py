# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

from .torch import Exponential
from .torch_distribution import TorchDistribution


def _log_binomial_coefficient(total, part):
    log_numer = (total + 1).lgamma()
    log_denom = (part + 1).lgamma() + (total - part + 1).lgamma()
    return log_numer - log_denom


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
        two = total.new_full((), 2)
        return _log_binomial_coefficient(remaining, two)

    @lazy_property
    def event_mask(self):
        return self.num_leaves.unsqueeze(-1) - torch.arange(self.event_shape[0]) >= 2

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("TODO")

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


class CoalescentTimesDiscretized(TorchDistribution):
    """
    Approximate distribution over discretized coalescent times in Kingman's
    ``n``-coalescent process [1,2].

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
    :param torch.Tensor time_steps: A series of time steps "dt" encoding a
        nonuniform discretization of the time axis.
    """
    has_rsample = True
    arg_constraints = {"num_leaves": constraints.positive_integer}
    support = constraints.positive  # partial constraint

    def __init__(self, num_leaves, time_steps, *, validate_args=None):
        self.num_leaves, _ = broadcast_all(num_leaves, time_steps[..., 0])
        batch_shape = self.num_leaves.shape
        event_shape = time_steps.shape[-1:]
        self.time_steps = time_steps.expand(batch_shape + event_shape)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("TODO")

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        raise NotImplementedError("TODO")

    def rsample(self, sample_shape):
        d = CoalescentIntervals(self.num_leaves)
        intervals = d.rsample(sample_shape)
        times = intervals.cumsum(-1)
        raise NotImplementedError("TODO")
