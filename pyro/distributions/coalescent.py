# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import weakref
from collections import namedtuple

import torch
from torch.distributions import constraints

from pyro.distributions.util import broadcast_shape
from pyro.ops.tensor_utils import safe_log

from .torch import Exponential
from .torch_distribution import TorchDistribution


class CoalescentTimesConstraint(constraints.Constraint):
    def __init__(self, leaf_times):
        self.leaf_times = leaf_times

    def check(self, value):
        # The only constraint is that there is always at least one lineage.
        coal_times = value
        phylogeny = _make_phylogeny(self.leaf_times, coal_times)
        return (phylogeny.lineages > 0).all(dim=-1)


class CoalescentTimes(TorchDistribution):
    """
    Distribution over coalescent times given irregular sampled ``leaf_times``.

    Sample values will be unordered sets of binary coalescent times. Each
    sample ``value`` will have cardinality ``value.size(-1) =
    leaf_times.size(-1) - 1``, so that phylogenies are complete binary trees.
    This distribution can thus be batched over multiple samples of phylogenies
    given fixed (number of) leaf times, e.g. over phylogeny samples from BEAST
    or MrBayes.

    **References**

    [1] J.F.C. Kingman (1982)
        "On the Genealogy of Large Populations"
        Journal of Applied Probability
    [2] J.F.C. Kingman (1982)
        "The Coalescent"
        Stochastic Processes and their Applications

    :param torch.Tensor leaf_times: Vector of times of sampling events, i.e.
        leaf nodes in the phylogeny. These can be arbitrary real numbers with
        arbitrary order and duplicates.
    """
    arg_constraints = {"leaf_times": constraints.real}

    def __init__(self, leaf_times, *, validate_args=None):
        event_shape = (leaf_times.size(-1) - 1,)
        batch_shape = leaf_times.shape[:-1]
        self.leaf_times = leaf_times
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return CoalescentTimesConstraint(self.leaf_times)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        coal_times = value
        phylogeny = _make_phylogeny(self.leaf_times, coal_times)

        # The coalescent process is like a Poisson process with rate binomial
        # in the number of lineages, which changes at each event.
        binomial = phylogeny.binomial[..., :-1]
        interval = phylogeny.times[..., :-1] - phylogeny.times[..., 1:]
        cumsum = (binomial * interval).cumsum(dim=-1)
        index = torch.nn.functional.pad(phylogeny.coal_index - 1, (1, 0), value=0)
        integral = cumsum.gather(-1, index)
        u = integral[..., 1:] - integral[..., :-1]
        log_prob = Exponential(1.).log_prob(u).sum(-1)

        # Scaling by those rates and accounting for log|jacobian|, the density
        # is that of a collection of independent Exponential intervals.
        log_abs_det_jacobian = phylogeny.coal_binomial.log().sum(-1).neg()
        return log_prob - log_abs_det_jacobian

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)[:-1]
        leaf_times = self.leaf_times.expand(shape + (-1,))
        return _sample_coalescent_times(leaf_times)


class CoalescentTimesWithRate(TorchDistribution):
    """
    Distribution over coalescent times given irregular sampled ``leaf_times``
    and piecewise constant coalescent rates defined on a regular time grid.

    This assumes a piecewise constant base coalescent rate specified on time
    intervals ``(-inf,1]``, ``[1,2]``, ..., ``[T-1,inf)``,  where ``T =
    rate_grid.size(-1)``. Leaves may be sampled at arbitrary real times, but
    are commonly sampled in the interval ``[0, T]``.

    Sample values will be unordered sets of binary coalescent times. Each
    sample ``value`` will have cardinality ``value.size(-1) =
    leaf_times.size(-1) - 1``, so that phylogenies are complete binary trees.
    This distribution can thus be batched over multiple samples of phylogenies
    given fixed (number of) leaf times, e.g. over phylogeny samples from BEAST
    or MrBayes.

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

    def __init__(self, leaf_times, rate_grid, *, validate_args=None):
        if leaf_times.dim() != 1:
            raise ValueError("leaf_times does not support batching")
        event_shape = (leaf_times.size(-1) - 1,)
        batch_shape = rate_grid.shape[:-1]
        self.leaf_times = leaf_times
        self.rate_grid = rate_grid
        self._unbroadcasted_rate_grid = rate_grid
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return CoalescentTimesConstraint(self.leaf_times)

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

        This is differentiable wrt ``rate_grid`` but neither ``leaf_times`` nor
        ``value = coal_times``.

        :param torch.Tensor value: A tensor of coalescent times. These denote
            sets of size ``leaf_times.size(-1) - 1`` along the trailing
            dimension and can be unordered along that dimension.
        :returns: Likelihood ``p(coal_times | leaf_times, rate_grid)``
        :rtype: torch.Tensor
        """
        if self._validate_args:
            self._validate_sample(value)
        coal_times = value
        phylogeny = _make_phylogeny(self.leaf_times, coal_times)

        # Compute survival factors for closed intervals.
        cumsum = self._unbroadcasted_rate_grid.cumsum(-1)
        cumsum = torch.nn.functional.pad(cumsum, (1, 0), value=0)
        integral = _interpolate(cumsum, phylogeny.times[..., 1:])  # ignore the final lonely leaf
        integral = integral[..., :-1] - integral[..., 1:]
        integral = integral.clamp(min=torch.finfo(integral.dtype).tiny)  # avoid nan
        log_prob = (phylogeny.binomial[..., 1:-1] * integral).sum(-1)

        # Compute density of coalescent events.
        i = coal_times.floor().clamp(min=0).long()
        rates = phylogeny.coal_binomial * _gather(self._unbroadcasted_rate_grid, -1, i)
        log_prob = log_prob + safe_log(rates).sum(-1)

        return log_prob


def _gather(tensor, dim, index):
    """
    Like :func:`torch.gather` but broadcasts.
    """
    if dim != -1:
        raise NotImplementedError
    shape = broadcast_shape(tensor.shape[:-1], index.shape[:-1]) + (-1,)
    tensor = tensor.expand(shape)
    index = index.expand(shape)
    return tensor.gather(dim, index)


def _interpolate(array, x):
    """
    Continuously index into the rightmost dim of an array, linearly
    interpolating between array values.
    """
    with torch.no_grad():
        x0 = x.floor().clamp(min=0, max=array.size(-1) - 2)
        x1 = x0 + 1
    f0 = _gather(array, -1, x0.long())
    f1 = _gather(array, -1, x1.long())
    return f0 * (x1 - x) + f1 * (x - x0)


def _weak_memoize(fn):
    cache = {}

    @functools.wraps(fn)
    def memoized_fn(*args):
        key = tuple(map(id, args))
        if key not in cache:
            cache[key] = fn(*args)
            for arg in args:
                weakref.finalize(arg, cache.pop, key, None)
        return cache[key]

    return memoized_fn


# This helper data structure has only timing information.
_Phylogeny = namedtuple("_Phylogeny", (
    "times", "signs", "lineages", "binomial", "coal_index", "coal_binomial",
))


@_weak_memoize
@torch.no_grad()
def _make_phylogeny(leaf_times, coal_times):
    assert leaf_times.size(-1) == 1 + coal_times.size(-1)
    assert not leaf_times.requires_grad
    assert not coal_times.requires_grad

    # Expand shapes to match.
    N = leaf_times.size(-1)
    batch_shape = broadcast_shape(leaf_times.shape[:-1], coal_times.shape[:-1])
    if leaf_times.shape[:-1] != batch_shape:
        leaf_times = leaf_times.expand(batch_shape + (N,))
    if coal_times.shape[:-1] != batch_shape:
        coal_times = coal_times.expand(batch_shape + (N - 1,))

    # Combine N sampling events (leaf_times) plus N-1 coalescent events
    # (coal_times) into a pair (times, signs) of arrays of length 2N-1, where
    # leaf sample sign is +1 and coalescent sign is -1.
    times = torch.cat([coal_times, leaf_times], dim=-1)
    signs = torch.linspace(1.5 - N, N - 0.5, 2 * N - 1).sign()  # e.g. [-1, -1, +1, +1, +1]

    # Sort the events reverse-ordered in time, i.e. latest to earliest.
    times, index = times.sort(dim=-1, descending=True)
    signs = signs[index]
    inv_index = index.new_empty(index.shape)
    inv_index.scatter_(-1, index, torch.arange(2 * N - 1).expand_as(index))

    # Compute the number n of lineages preceding each event, then the binomial
    # coefficients that will multiply the base coalescence rate.
    lineages = signs.cumsum(-1)
    binomial = lineages * (lineages - 1) / 2

    # Compute the binomial coefficient following each coalescent event.
    coal_index = inv_index[..., :N - 1]
    coal_binomial = binomial.gather(-1, coal_index - 1)
    assert (coal_binomial > 0).all()

    return _Phylogeny(times, signs, lineages, binomial, coal_index, coal_binomial)


@torch.no_grad()
def _sample_coalescent_times(leaf_times):
    N = leaf_times.size(-1)
    batch_shape = leaf_times.shape[:-1]

    # We don't bother to implement a version that vectorizes over batches;
    # instead we simply sequentially sample and stack.
    if batch_shape:
        flat_leaf_times = leaf_times.reshape(-1, N)
        flat_coal_times = torch.stack(list(map(_sample_coalescent_times, flat_leaf_times)))
        return flat_coal_times.reshape(batch_shape + (N - 1,))
    assert leaf_times.shape == (N,)

    # Sequentially sample coalescent events from latest to earliest.
    leaf_times = leaf_times.sort(dim=-1, descending=True).values
    coal_times = []
    # Start with the minimum of two active leaves.
    leaf = 1
    t = leaf_times[leaf]
    active = 2
    binomial = active * (active - 1) / 2
    for u in Exponential(1.).sample((N - 1,)):
        while leaf + 1 < N and u > (t - leaf_times[leaf + 1]) * binomial:
            # Move past the next leaf.
            leaf += 1
            u -= (t - leaf_times[leaf]) * binomial
            t = leaf_times[leaf]
            active += 1
            binomial = active * (active - 1) / 2
        # Add a coalescent event.
        t = t - u / binomial
        active -= 1
        binomial = active * (active - 1) / 2
        coal_times.append(t)

    return torch.stack(coal_times)
