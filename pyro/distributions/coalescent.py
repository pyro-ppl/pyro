# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import weakref
from collections import namedtuple

import torch
from torch.distributions import constraints

from pyro.distributions.util import broadcast_shape, is_validation_enabled
from pyro.ops.special import safe_log

from .torch_distribution import TorchDistribution


class CoalescentTimesConstraint(constraints.Constraint):
    def __init__(self, leaf_times, *, ordered=True):
        self.leaf_times = leaf_times
        self.ordered = ordered

    def check(self, value):
        # There must always at least one lineage.
        coal_times = value
        phylogeny = _make_phylogeny(self.leaf_times, coal_times)
        at_least_one_lineage = (phylogeny.lineages > 0).all(dim=-1)
        if not self.ordered:
            return at_least_one_lineage

        # Inputs must be ordered.
        ordered = (value[..., :-1] <= value[..., 1:]).all(dim=-1)
        return ordered & at_least_one_lineage


class CoalescentTimes(TorchDistribution):
    """
    Distribution over coalescent times given irregular sampled ``leaf_times``.

    Sample values will be sorted sets of binary coalescent times. Each sample
    ``value`` will have cardinality ``value.size(-1) = leaf_times.size(-1) -
    1``, so that phylogenies are complete binary trees.  This distribution can
    thus be batched over multiple samples of phylogenies given fixed (number
    of) leaf times, e.g. over phylogeny samples from BEAST or MrBayes.

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
        log_prob = -(binomial * interval).sum(-1)

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

    Sample values will be sorted sets of binary coalescent times. Each sample
    ``value`` will have cardinality ``value.size(-1) = leaf_times.size(-1) -
    1``, so that phylogenies are complete binary trees.  This distribution can
    thus be batched over multiple samples of phylogenies given fixed (number
    of) leaf times, e.g. over phylogeny samples from BEAST or MrBayes.

    This distribution implements :meth:`log_prob` but not ``.sample()``.

    See also :class:`~pyro.distributions.CoalescentRateLikelihood`.

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

    :param torch.Tensor leaf_times: Tensor of times of sampling events, i.e.
        leaf nodes in the phylogeny. These can be arbitrary real numbers with
        arbitrary order and duplicates.
    :param torch.Tensor rate_grid: Tensor of base coalescent rates (pairwise
        rate of coalescence). For example in a simple SIR model this might be
        ``beta S / I``. The rightmost dimension is time, and this tensor
        represents a (batch of) rates that are piecwise constant in time.
    """
    arg_constraints = {"leaf_times": constraints.real,
                       "rate_grid": constraints.positive}

    def __init__(self, leaf_times, rate_grid, *, validate_args=None):
        batch_shape = broadcast_shape(leaf_times.shape[:-1], rate_grid.shape[:-1])
        event_shape = (leaf_times.size(-1) - 1,)
        self.leaf_times = leaf_times
        self.rate_grid = rate_grid
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return CoalescentTimesConstraint(self.leaf_times)

    @property
    def duration(self):
        return self.rate_grid.size(-1)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CoalescentTimesWithRate, _instance)
        new.leaf_times = self.leaf_times
        new.rate_grid = self.rate_grid
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
            dimension and should be sorted along that dimension.
        :returns: Likelihood ``p(coal_times | leaf_times, rate_grid)``
        :rtype: torch.Tensor
        """
        if self._validate_args:
            self._validate_sample(value)
        coal_times = value
        phylogeny = _make_phylogeny(self.leaf_times, coal_times)

        # Compute survival factors for closed intervals.
        cumsum = self.rate_grid.cumsum(-1)
        cumsum = torch.nn.functional.pad(cumsum, (1, 0), value=0)
        integral = _interpolate_gather(cumsum, phylogeny.times[..., 1:])  # ignore the final lonely leaf
        integral = integral[..., :-1] - integral[..., 1:]
        integral = integral.clamp(min=torch.finfo(integral.dtype).tiny)  # avoid nan
        log_prob = -(phylogeny.binomial[..., 1:-1] * integral).sum(-1)

        # Compute density of coalescent events.
        i = coal_times.floor().clamp(min=0, max=self.duration - 1).long()
        rates = phylogeny.coal_binomial * _gather(self.rate_grid, -1, i)
        log_prob = log_prob + safe_log(rates).sum(-1)

        batch_shape = broadcast_shape(self.batch_shape, value.shape[:-1])
        log_prob = log_prob.expand(batch_shape)
        return log_prob


class CoalescentRateLikelihood:
    """
    EXPERIMENTAL This is not a :class:`~pyro.distributions.Distribution`, but
    acts as a transposed version of :class:`CoalescentTimesWithRate` making the
    elements of ``rate_grid`` independent and thus compatible with ``plate``
    and ``poutine.markov``. For non-batched inputs the following are all
    equivalent likelihoods::

        # Version 1.
        pyro.sample("coalescent",
                    CoalescentTimesWithRate(leaf_times, rate_grid),
                    obs=coal_times)

        # Version 2. using pyro.plate
        likelihood = CoalescentRateLikelihood(leaf_times, coal_times, len(rate_grid))
        with pyro.plate("time", len(rate_grid)):
            pyro.factor("coalescent", likelihood(rate_grid))

        # Version 3. using pyro.markov
        likelihood = CoalescentRateLikelihood(leaf_times, coal_times, len(rate_grid))
        for t in pyro.markov(range(len(rate_grid))):
            pyro.factor("coalescent_{}".format(t), likelihood(rate_grid[t], t))

    The third version is useful for e.g.
    :class:`~pyro.infer.smcfilter.SMCFilter` where ``rate_grid`` might be
    computed sequentially.

    :param torch.Tensor leaf_times: Tensor of times of sampling events, i.e.
        leaf nodes in the phylogeny. These can be arbitrary real numbers with
        arbitrary order and duplicates.
    :param torch.Tensor coal_times: A tensor of coalescent times. These denote
        sets of size ``leaf_times.size(-1) - 1`` along the trailing dimension
        and should be sorted along that dimension.
    :param int duration: Size of the rate grid, ``rate_grid.size(-1)``.
    """
    def __init__(self, leaf_times, coal_times, duration, *, validate_args=None):
        assert leaf_times.size(-1) == 1 + coal_times.size(-1)
        assert isinstance(duration, int) and duration >= 2
        if validate_args is True or validate_args is None and is_validation_enabled:
            constraint = CoalescentTimesConstraint(leaf_times, ordered=False)
            if not constraint.check(coal_times).all():
                raise ValueError("Invalid (leaf_times, coal_times)")

        phylogeny = _make_phylogeny(leaf_times, coal_times)
        batch_shape = phylogeny.times.shape[:-1]
        new_zeros = leaf_times.new_zeros
        new_ones = leaf_times.new_ones

        # Construct linear part from intervals of survival outside of [0,duration].
        times = phylogeny.times.clamp(max=0)
        intervals = times[..., 1:] - times[..., :-1]
        pre_linear = (phylogeny.binomial[..., :-1] * intervals).sum(-1, keepdim=True)
        times = phylogeny.times.clamp(min=duration)
        intervals = times[..., 1:] - times[..., :-1]
        post_linear = (phylogeny.binomial[..., :-1] * intervals).sum(-1, keepdim=True)
        self._linear = torch.cat([pre_linear,
                                  new_zeros(pre_linear.shape[:-1] + (duration - 2,)),
                                  post_linear], dim=-1)

        # Construct linear part from intervals of survival within [0, duration].
        times = phylogeny.times.clamp(min=0, max=duration)
        sparse_diff = phylogeny.binomial[..., :-1] - phylogeny.binomial[..., 1:]
        dense_diff = new_zeros(batch_shape + (1 + duration,))
        _interpolate_scatter_add_(dense_diff, times[..., 1:], sparse_diff)
        self._linear += dense_diff.flip([-1]).cumsum(-1)[..., :-1].flip([-1])

        # Construct const and log part from coalescent events.
        coal_index = coal_times.floor().clamp(min=0, max=duration - 1).long()
        self._const = new_zeros(batch_shape + (duration,))
        self._const.scatter_add_(-1, coal_index, phylogeny.coal_binomial.log())
        self._log = new_zeros(batch_shape + (duration,))
        self._log.scatter_add_(-1, coal_index, new_ones(coal_index.shape))

    def __call__(self, rate_grid, t=slice(None)):
        """
        Computes the likelihood of [1] equations 7-9 for one or all time
        points.

        **References**

        [1] A. Popinga, T. Vaughan, T. Statler, A.J. Drummond (2014)
            "Inferring epidemiological dynamics with Bayesian coalescent
            inference: The merits of deterministic and stochastic models"
            https://arxiv.org/pdf/1407.1792.pdf

        :param torch.Tensor rate_grid: Tensor of base coalescent rates
            (pairwise rate of coalescence). For example in a simple SIR model
            this might be ``beta S / I``. The rightmost dimension is time, and
            this tensor represents a (batch of) rates that are piecwise
            constant in time.
        :param time: Optional time index by which the input was sliced, as in
            ``rate_grid[..., t]`` This can be an integer for sequential models
            or ``slice(None)`` for vectorized models.
        :type time: int or slice
        :returns: Likelihood ``p(coal_times | leaf_times, rate_grid)``,
            or a part of that likelihood corresponding to a single time step.
        :rtype: torch.Tensor
        """
        const = self._const[..., t]
        linear = self._linear[..., t] * rate_grid
        log = self._log[..., t] * rate_grid.clamp(min=torch.finfo(rate_grid.dtype).tiny).log()
        return const + linear + log


def bio_phylo_to_times(tree, *, get_time=None):
    """
    Extracts coalescent summary statistics from a phylogeny, suitable for use
    with :class:`~pyro.distributions.CoalescentRateLikelihood`.

    :param Bio.Phylo.BaseTree.Clade tree: A phylogenetic tree.
    :param callable get_time: Optional function to extract the time point of
        each sub-:class:`~Bio.Phylo.BaseTree.Clade`. If absent, times will be
        computed by cumulative `.branch_length`.
    :returns: A pair of :class:`~torch.Tensor` s ``(leaf_times, coal_times)``
        where ``leaf_times`` are times of sampling events (leaf nodes in the
        phylogenetic tree) and ``coal_times`` are times of coalescences (leaf
        nodes in the phylogenetic binary tree).
    :rtype: tuple
    """
    if get_time is None:
        # Compute time as cumulative branch length.
        def get_branch_length(clade):
            branch_length = clade.branch_length
            return 1.0 if branch_length is None else branch_length
        times = {tree.root: get_branch_length(tree.root)}

    leaf_times = []
    coal_times = []
    for clade in tree.find_clades():
        if get_time is None:
            time = times[clade]
            for child in clade:
                times[child] = time + get_branch_length(child)
        else:
            time = get_time(clade)

        num_children = len(clade)
        if num_children == 0:
            leaf_times.append(time)
        else:
            # Pyro expects binary coalescent events, so we split n-ary events
            # into n-1 separate binary events.
            for _ in range(num_children - 1):
                coal_times.append(time)
    assert len(leaf_times) == 1 + len(coal_times)

    leaf_times = torch.tensor(leaf_times)
    coal_times = torch.tensor(coal_times)
    return leaf_times, coal_times


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


def _interpolate_gather(array, x):
    """
    Like ``torch.gather(-1, array, x)`` but continuously indexes into the
    rightmost dim of an array, linearly interpolating between array values.
    """
    with torch.no_grad():
        x0 = x.floor().clamp(min=0, max=array.size(-1) - 2)
        x1 = x0 + 1
    f0 = _gather(array, -1, x0.long())
    f1 = _gather(array, -1, x1.long())
    return f0 * (x1 - x) + f1 * (x - x0)


def _interpolate_scatter_add_(dst, x, src):
    """
    Like ``dst.scatter_add_(-1, x, src)`` but continuously index into the
    rightmost dim of an array, linearly interpolating between array values.
    """
    with torch.no_grad():
        x0 = x.floor().clamp(min=0, max=dst.size(-1) - 2)
        x1 = x0 + 1
    dst.scatter_add_(-1, x0.long(), src * (x1 - x))
    dst.scatter_add_(-1, x1.long(), src * (x - x0))
    return dst


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
    "times", "signs", "lineages", "binomial", "coal_binomial",
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

    return _Phylogeny(times, signs, lineages, binomial, coal_binomial)


def _sample_coalescent_times(leaf_times):
    leaf_times = leaf_times.detach()
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
    leaf_times = leaf_times.sort(dim=-1, descending=True).values.tolist()
    coal_times = []
    # Start with the minimum of two active leaves.
    leaf = 1
    t = leaf_times[leaf]
    active = 2
    binomial = active * (active - 1) / 2
    for u in torch.empty(N - 1).exponential_().tolist():
        while leaf + 1 < N and u > (t - leaf_times[leaf + 1]) * binomial:
            # Move past the next leaf.
            leaf += 1
            u -= (t - leaf_times[leaf]) * binomial
            t = leaf_times[leaf]
            active += 1
            binomial = active * (active - 1) / 2
        # Add a coalescent event.
        t -= u / binomial
        active -= 1
        binomial = active * (active - 1) / 2
        coal_times.append(t)
    coal_times.reverse()

    return torch.tensor(coal_times)
