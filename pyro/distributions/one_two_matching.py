# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.util import warn_if_inf, warn_if_nan

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
        if sample_shape:
            return torch.stack([self.sample(sample_shape[1:])
                                for _ in range(sample_shape[0])])
        # Consider initializing with heuristic or MAP
        # https://arxiv.org/pdf/0709.1190
        # http://proceedings.mlr.press/v15/huang11a/huang11a.pdf
        # followed by a small number of MCMC steps
        # http://www.cs.toronto.edu/~mvolkovs/nips2012_sampling.pdf
        raise NotImplementedError


def log_count_one_two_matchings(logits, bp_iters):
    # This adapts [1] from 1-1 matchings to 1-2 matchings.
    #
    # The core difference is that the two destination assignments are sampled
    # without replacement, thereby following a multivariate Wallenius'
    # noncentral hypergeometric distribution [2]; this results in a change to
    # the destin->source messages m_ds, relative to [1].
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

    def clamp(x):
        # We clamp generously so results can be added without overflow.
        x.data.clamp_(min=finfo.min / 4, max=finfo.max / 4)
        return x

    def log(x):
        x.data.clamp_(min=finfo.eps, max=finfo.max)
        return x.log()

    # Perform belief propagation, adapting [1] Lemma 29 to use log-space
    # representations of potentials f, messages m, and beliefs b. Local
    # marginals z and local partition functions Z are in linear space.
    f = clamp(logits / 2)  # Split potential in half between source & destin.
    m_ds = torch.zeros_like(f)
    for i in range(bp_iters):
        # Update source->destin messages by marginalizing over a simple
        # categorical distribution.
        b = f + m_ds
        b.data -= b.data.max(-1, True).values
        z = b.exp()
        Z = z.sum(-1, True)
        m_sd = clamp(b - log(Z - z) - m_ds)
        warn_if_nan(m_sd, "m_sd iter {}".format(i))

        # Update source->destin messages by marginalizing over the
        # distribution of weighted unordered pairs without replacement.
        b = f + m_sd
        shift = b.data.max(-2).values
        b.data -= shift
        z = b.exp()
        Z = z.sum(-2)
        z2 = z * (Z - z)  # "choose this and any other source"
        Z2 = z2.sum(-2) / 2  # "ordered pairs, modulo order"
        m_ds = clamp(log(z2) - log(Z2 - z2) - m_sd)
        warn_if_nan(m_ds, "m_ds iter {}".format(i))

    # Evaluate the pseudo-dual Bethe free energy, adapting [1] Lemma 31.
    energy_d = log(Z2).sum() + 2 * shift.sum()  # destin->source pairs.
    energy_s = (f + m_ds).logsumexp(-1).sum()  # source->destin Categoricals.
    energy_ds = (m_sd + m_ds).exp().log1p().sum()  # Bernoullis for each edge.
    energy = energy_ds - energy_d - energy_s
    warn_if_nan(energy, "energy")
    warn_if_inf(energy, "energy")
    return -energy


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
