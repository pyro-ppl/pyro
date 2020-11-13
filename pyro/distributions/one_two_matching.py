# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.util import warn_if_inf, warn_if_nan

from .torch import Categorical
from .torch_distribution import TorchDistribution


class OneTwoMatchingConstraint(constraints.Constraint):
    def __init__(self, num_destins):
        self.num_destins = num_destins
        self.num_sources = 2 * num_destins

    def check(self, value):
        if value.dim() == 0:
            warnings.warn("Invalid event_shape: ()")
            return torch.tensor(False)
        batch_shape, event_shape = value.shape[:-1], value.shape[-1:]
        if event_shape != (self.num_sources,):
            warnings.warn("Invalid event_shape: {}".format(event_shape))
            return torch.tensor(False)
        if value.min() < 0 or value.max() >= self.num_destins:
            warnings.warn("Value out of bounds")
            return torch.tensor(False)
        counts = torch.zeros(batch_shape + (self.num_destins,))
        counts.scatter_add_(-1, value, torch.ones(value.shape))
        if (counts != 2).any():
            warnings.warn("Matching is not binary")
            return torch.tensor(False)
        return torch.tensor(True)


class OneTwoMatching(TorchDistribution):
    r"""
    Random matching from ``2*N`` sources to ``N`` destinations where each
    source matches exactly **one** destination and each destination matches
    exactly **two** sources. Samples are represented as long tensors of shape
    ``(2*N,)`` taking values in ``{0,...,N-1}`` and satisfying the above
    one-two constraint. The log probability of a sample ``v`` is the sum of
    edge logits, up to the log partition function ``log Z``:

    .. math::

        \log p(v) = \sum_s \text{logits}[s, v[s]] - \log Z

    The :meth:`log_partition_function` and :meth:`log_prob` methods use a Bethe
    approximation [1,2,3]. This currently does not implement :meth:`sample`.

    **References:**

    [1] Michael Chertkov, Lukas Kroc, Massimo Vergassola (2008)
        "Belief propagation and beyond for particle tracking"
        https://arxiv.org/pdf/0806.1199.pdf
    [2] Bert Huang, Tony Jebara (2009)
        "Approximating the Permanent with Belief Propagation"
        https://arxiv.org/pdf/0908.1769.pdf
    [3] Pascal O. Vontobel (2012)
        "The Bethe Permanent of a Non-Negative Matrix"
        https://arxiv.org/pdf/1107.4196.pdf

    :param Tensor logits: An ``(2 * N, N)``-shaped tensor of edge logits.
    :param int bp_iters: Optional number of belief propagation iterations. If
        unspecified or ``None`` an expensive exact algorithm will be used.
    """
    arg_constraints = {"logits": constraints.real}
    has_enumerate_support = True

    def __init__(self, logits, *, bp_iters=None, validate_args=None):
        if logits.dim() != 2:
            raise NotImplementedError("OneTwoMatching does not support batching")
        assert bp_iters is None or isinstance(bp_iters, int) and bp_iters > 0
        self.num_sources, self.num_destins = logits.shape
        assert self.num_sources == 2 * self.num_destins
        self.logits = logits
        batch_shape = ()
        event_shape = (self.num_sources,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self.bp_iters = bp_iters

    @constraints.dependent_property
    def support(self):
        return OneTwoMatchingConstraint(self.num_destins)

    @lazy_property
    def log_partition_function(self):
        if self.bp_iters is not None:
            return log_count_one_two_matchings(self.logits, self.bp_iters)
        # Brute force.
        d = self.enumerate_support()
        s = torch.arange(d.size(-1), dtype=d.dtype, device=d.device)
        return self.logits[s, d].sum(-1).logsumexp(-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        d = value
        s = torch.arange(d.size(-1), dtype=d.dtype, device=d.device)
        return self.logits[s, d].sum(-1) - self.log_partition_function

    def enumerate_support(self, expand=True):
        return enumerate_one_two_matchings(self.num_destins)

    def sample(self, sample_shape=torch.Size()):
        if self.bp_iters is None:
            # Brute force.
            d = self.enumerate_support()
            s = torch.arange(d.size(-1), dtype=d.dtype, device=d.device)
            logits = self.logits[s, d].sum(-1)
            sample = Categorical(logits=logits).sample(sample_shape)
            return d[sample]

        if sample_shape:
            return torch.stack([self.sample(sample_shape[1:])
                                for _ in range(sample_shape[0])])
        # Consider initializing with heuristic or MAP
        # https://arxiv.org/pdf/0709.1190
        # http://proceedings.mlr.press/v15/huang11a/huang11a.pdf
        # followed by a small number of MCMC steps
        # http://www.cs.toronto.edu/~mvolkovs/nips2012_sampling.pdf
        raise NotImplementedError


def log_count_one_two_matchings_v1(logits, bp_iters):
    # This adapts [1] from 1-1 matchings to 1-2 matchings.
    #
    # The core difference is that the two destination assignments are sampled
    # without replacement, thereby following a multivariate Wallenius'
    # noncentral hypergeometric distribution [2]; this results in a change to
    # the destin -> source messages m_ds, relative to [1].
    #
    # [1] Pascal O. Vontobel (2012)
    #     "The Bethe Permanent of a Non-Negative Matrix"
    #     https://arxiv.org/pdf/1107.4196.pdf
    # [2] https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution
    assert bp_iters > 0
    assert logits.dim() == 2
    num_sources, num_destins = logits.shape
    assert num_sources == 2 * num_destins
    finfo = torch.finfo(logits.dtype)
    shift = logits.data.max(1, True).values
    p = (logits - shift).exp().clamp(min=finfo.eps)

    # Perform loopy belief propagation, adapting [1] Lemma 29.
    m_ds = torch.ones_like(p)
    for i in range(bp_iters):
        # Update source -> destin messages by marginalizing over a simple
        # categorical distribution.
        z = p * m_ds
        Z = z.sum(-1, True)
        z_ = (Z - z).clamp(min=finfo.eps)
        m_sd = z / (z_ * m_ds)
        m_sd = m_sd.clamp(min=finfo.eps, max=finfo.max)

        # Update destin -> source messages by marginalizing over the
        # distribution of weighted unordered pairs without replacement.
        z = m_sd
        Z = z.sum(-2)
        z2 = z * (Z - z).clamp(min=finfo.eps)  # "choose this and any other source"
        Z2 = z2.sum(-2) / 2  # "ordered pairs, modulo order"
        z2_ = (Z2 - z2).clamp(min=finfo.eps)
        m_ds = z2 / (z2_ * m_sd)
        m_ds = m_ds.clamp(min=finfo.eps, max=finfo.max)

    # # Evaluate the pseudo-dual Bethe free energy, adapting [1] Lemma 31.
    energy_d = Z2.log().sum()  # destin->source pairs.
    energy_s = (p * m_ds).sum(-1).log().sum()  # source->destin Categoricals.
    energy_ds = (m_sd * m_ds).log1p().sum()  # Bernoullis for each edge.
    energy = energy_ds - energy_d - energy_s
    warn_if_nan(energy, "energy")
    warn_if_inf(energy, "energy")
    return shift.sum() - energy


def log_count_one_two_matchings_v2(logits, bp_iters, *, mf_iters=10, rate=0.5):
    # This adapts [1] from 1-1 matchings to 1-2 matchings.
    #
    # [1] M Chertkov, AB Yedidia (2013)
    #     "Approximating the permanent with fractional belief propagation"
    #     http://www.jmlr.org/papers/volume14/chertkov13a/chertkov13a.pdf
    assert logits.dim() == 2
    num_sources, num_destins = logits.shape
    assert num_sources == 2 * num_destins
    finfo = torch.finfo(logits.dtype)
    shift = logits.data.max(1, True).values
    p = (logits - shift).exp().clamp(min=finfo.eps)

    # Initialize via Fermi mean field, adapting [1] Eqns 33-34.
    with torch.no_grad():
        u = p.new_ones(num_sources, 1)
        v = p.new_ones(1, num_destins)
        for _ in range(mf_iters):
            b = p / (p + u * v)
            u = u * b.sum(1, True)
            v = v * b.sum(0) / 2

    # Perform Bethe loopy belief propagation, adapting [1] Eqns 29-30.
    for _ in range(bp_iters):
        bb = b * b
        u_new = (p / v).sum(1, True) / (1 - bb.sum(1, True)).clamp(min=finfo.eps)
        v_new = (p / u).sum(0) / (2 - bb.sum(0)).clamp(min=finfo.eps)
        b_new = p / (p + (b.sum(0) / 4 + b.sum(1, True) / 2 - b).pow(2) * (u * v))
        u, v = u_new, v_new
        b = rate * b + (1 - rate) * b_new
        # Perform a Sinkhorn update.
        b = b / b.sum(0)
        b = b / b.sum(1, True)

    # Evaluate the Bethe free energy [1] Eqn 4.
    b = b.clamp(min=finfo.tiny)
    b_ = (1 - b).clamp(min=finfo.tiny)
    energy = (b * (b / p).log() - b_ * b_.log()).sum()
    warn_if_nan(energy, "energy")
    warn_if_inf(energy, "energy")
    return shift.sum() - energy


def log_count_one_two_matchings_v3(logits, bp_iters, *, temperature=0.5):
    # This implements a mean field approximation via Sinkhorn iteration.
    #
    # [1] M Chertkov, AB Yedidia (2013)
    #     "Approximating the permanent with fractional belief propagation"
    #     http://www.jmlr.org/papers/volume14/chertkov13a/chertkov13a.pdf
    assert logits.dim() == 2
    num_sources, num_destins = logits.shape
    assert num_sources == 2 * num_destins
    finfo = torch.finfo(logits.dtype)
    shift = logits.data.max(1, True).values
    p = (logits - shift).exp().clamp(min=finfo.tiny)

    # Approximate mean field beliefs b via Sinkhorn iteration.
    b = p / p.sum(1, True)
    for _ in range(bp_iters):
        b /= b.sum(0)
        b /= b.sum(1, True)

    # Evaluate the free energy [1] Eqn 9.
    b = b.clamp(min=finfo.tiny)
    b_ = (1 - b).clamp(min=finfo.tiny)
    energy = (b * (b / p).log() - temperature * b_ * b_.log()).sum()
    warn_if_nan(energy, "energy")
    warn_if_inf(energy, "energy")
    return shift.sum() - energy


log_count_one_two_matchings = log_count_one_two_matchings_v3


def enumerate_one_two_matchings(num_destins):
    if num_destins == 1:
        return torch.tensor([[0, 0]])

    num_sources = num_destins * 2
    subproblem = enumerate_one_two_matchings(num_destins - 1)
    subsize = subproblem.size(0)
    result = torch.empty(subsize * num_sources * (num_sources - 1) // 2,
                         num_sources,
                         dtype=torch.long)

    # Iterate over pairs of sources s0<s1 matching the last destination d.
    d = num_destins - 1
    pos = 0
    for s1 in range(num_sources):
        for s0 in range(s1):
            block = result[pos:pos+subsize]
            block[:, :s0] = subproblem[:, :s0]
            block[:, s0] = d
            block[:, s0 + 1:s1] = subproblem[:, s0:s1 - 1]
            block[:, s1] = d
            block[:, s1 + 1:] = subproblem[:, s1 - 1:]
            pos += subsize
    return result
