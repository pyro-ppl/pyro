# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
from weakref import WeakKeyDictionary

import torch
from torch.distributions import constraints

from pyro.ops.indexing import Vindex
from pyro.ops.tensor_utils import safe_log

from .torch_distribution import TorchDistribution


class CoalescentTimesWithRate(TorchDistribution):
    """
    Distribution over coalescent times given irregular sampled ``leaf_times``
    and piecewise constant coalescent rates defined on a regular time grid.

    This assumes a piecewise constant rate specified on time intervals
    ``(-inf,1]``, ``[1,2]``, ..., ``[T-1,T]``,  where ``T =
    rate_grid.size(-1)``.  Leaves may be sampled at arbitrary times in the real
    interval ``(-inf, T]``.

    Sample values will be unordered sets of binary coalescent times in the
    interval ``(-inf, T]``, where ``T = rate_grid.size(-1)``. Each sample
    ``value`` will have cardinality ``value.size(-1) = leaf_times.size(-1) -
    1``, so that phylogenies are complete binary trees. This distribution can
    thus be batched over multiple samples of phylogenies given fixed (number
    of) leaf times, e.g. over phylogeny samples from BEAST or MrBayes.

    This distribution implements :meth:`log_prob` but not ``.sample()``.

    **References**

    [1] J.F.C. Kingman (1982)
        "On the Genealogy of Large Populations"
        Journal of Applied Probability
    [2] J.F.C. Kingman (1982)
        "The Coalescent"
        Stochastic Processes and their Applications
    [3] A. Popinga, T. Vaughan, T. Statler, A.J. Drummond (2014)
        "Inferring epidemiological dynamics with Bayesian coalescent inference:
        The merits of deterministic and stochastic models"
        https://arxiv.org/pdf/1407.1792.pdf

    :param torch.Tensor leaf_times:
    :param torch.Tensor rate_grid:
    """
    arg_constraints = {"leaf_times": constraints.real,
                       "rate_grid": constraints.positive}
    support = constraints.real  # partial constraint

    def __init__(self, leaf_times, rate_grid, *, validate_args=None):
        if leaf_times.dim() != 1:
            raise ValueError("leaf_times does not support batching")
        event_shape = (leaf_times.size(-1) - 1,)
        batch_shape = rate_grid.shape[:-1]
        self.leaf_times = leaf_times
        self.rate_grid = rate_grid
        self._unbroadcasted_rate_grid = rate_grid
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CoalescentTimesWithRate, _instance)
        new.leaf_times = self.leaf_times
        new.rate_grid = self.rate_grid.expand(batch_shape + (-1,))
        new._unbroadcasted_rate_grid = self._unbroadcasted_rate_grid
        super(CoalescentTimesWithRate, new).__init__(
            batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get("_validate_args")
        return new

    def log_prob(self, value):
        """
        Computes likelihood as in equations 7-8 of [3].

        This has time complexity ``O(T + S N log(N))`` where ``T`` is the
        number of time steps, ``N`` is the number of leaves, and ``S =
        sample_shape.numel()`` is the number of samples of ``value``.
        Additionally this caches ``(leaf_times, coal_times)`` pairs such that
        if only ``rate_grid`` varies across multiple calls, the amortized time
        complexity is ``O(T + S N)``.

        :param torch.Tensor value: A tensor of coalescent times ranging in
            ``(-inf,T)`` where ``T = self.rate_grid.size(-1)``. These denote
            sets of size ``leaf_times.size(-1) - 1`` along the trailing
            dimension and can be unordered along that dimension.
        :returns: Likelihood ``p(coal_times | leaf_times, rate_grid)``
        :rtype: torch.Tensor
        """
        if self._validate_args:
            self._validate_sample(value)
        coal_times = value
        times, binomial, coal_binomial = _preprocess(self.leaf_times, coal_times)

        # Compute survival factors for closed intervals.
        cumsum = self._unbroadcasted_rate_grid.cumsum(-1)
        cumsum = torch.nn.functional.pad(cumsum, (1, 0), value=0)
        integral = _interpolate(cumsum, times[..., 1:])  # ignore the first lonely leaf
        integral = integral[..., :-1] - integral[..., 1:]
        integral = integral.clamp(min=torch.finfo(integral.dtype).tiny)  # avoid nan
        log_prob = (binomial[..., 1:-1] * integral).sum(-1)

        # Compute density of coalescent events.
        i = coal_times.floor().clamp(min=0).long()
        rates = coal_binomial * self._unbroadcasted_rate_grid[..., i]
        log_prob = log_prob - safe_log(rates)

        return log_prob


def _interpolate(array, x):
    """
    Continuously index into the rightmost dim of an array, linearly
    interpolating between array values.
    """
    with torch.no_grad():
        x0 = x.floor().clamp(min=0, max=array.size(-1) - 2)
        x1 = x0 + 1
    f0 = Vindex(array)[..., x0.long()]
    f1 = Vindex(array)[..., x1.long()]
    return f0 * (x1 - x) + f1 * (x - x0)


def _weak_memoize_2(fn):
    cache = WeakKeyDictionary()

    @functools.wraps(fn)
    def memoized_fn(x, y):
        if x not in cache:
            cache[x] = WeakKeyDictionary()
            cache[x][y] = fn(x, y)
        elif y not in cache[x]:
            cache[x][y] = fn(x, y)
        return cache[x][y]

    return memoized_fn


@_weak_memoize_2
@torch.no_grad()
def _preprocess(leaf_times, coal_times):
    assert leaf_times.dim() == 1
    assert leaf_times.size(-1) == 1 + coal_times.size(-1)
    assert not leaf_times.requires_grad
    assert not coal_times.requires_grad

    # Expand leaf_times to match coal_times.
    N = leaf_times.size(-1)
    batch_shape = coal_times.shape[:-1]
    if batch_shape:
        leaf_times = leaf_times.expand(batch_shape + (N,))

    # Combine N sampling events (leaf_times) and N-1 coalescent events
    # (coal_times) into a pair (times, signs) of arrays of length 2N-1, where
    # leaf sample sign is +1 and coalescent sign is -1.
    times = torch.cat([coal_times, leaf_times], dim=-1)
    signs = torch.arange(1. - N, N).sign()  # e.g. [-1, -1, +1, +1, +1]

    # Sort the events reverse-ordered in time.
    times, index = times.sort(dim=-1, descending=True)
    signs = signs.index_select(-1, index)
    inv_index = index.new_empty(index.shape)
    inv_index[..., index] = torch.arange(2 * N - 1).expand_as(index)

    # Compute the number n of lineages preceding each event, then the binomial
    # coefficients that will adjust coalescence rate.
    n = signs.cumsum(-1)
    binomial = n * (n - 1) / 2

    # Compute the binomial coefficient following each coalescent event.
    coal_index = inv_index[..., :N - 1]
    coal_binomial = binomial.index_select(-1, coal_index + 1)

    return times, binomial, coal_binomial
