# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import warnings

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from .torch import Categorical
from .torch_distribution import TorchDistribution

logger = logging.getLogger(__name__)


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
    exactly **two** sources.

    Samples are represented as long tensors of shape ``(2*N,)`` taking
    values in ``{0,...,N-1}`` and satisfying the above one-two constraint.
    The log probability of a sample ``v`` is the sum of edge logits, up to
    the log partition function ``log Z``:

    .. math::

        \log p(v) = \sum_s \text{logits}[s, v[s]] - \log Z

    Exact computations are expensive. To enable tractable approximations, set a
    number of belief propagation iterations via the ``bp_iters`` argument.  The
    :meth:`log_partition_function` and :meth:`log_prob` methods use a Bethe
    approximation [1,2,3,4].

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
    [4] M Chertkov, AB Yedidia (2013)
        "Approximating the permanent with fractional belief propagation"
        http://www.jmlr.org/papers/volume14/chertkov13a/chertkov13a.pdf

    :param Tensor logits: An ``(2 * N, N)``-shaped tensor of edge logits.
    :param int bp_iters: Optional number of belief propagation iterations. If
        unspecified or ``None`` expensive exact algorithms will be used.
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
        if self.bp_iters is None:
            # Brute force.
            d = self.enumerate_support()
            s = torch.arange(d.size(-1), dtype=d.dtype, device=d.device)
            return self.logits[s, d].sum(-1).logsumexp(-1)

        # Approximate mean field beliefs b via Sinkhorn iteration.
        # We find that Sinkhorn iteration is more robust and faster than the
        # optimal belief propagation updates suggested in [1-4].
        finfo = torch.finfo(self.logits.dtype)
        # Note gradients are more accurate when shift is not detached.
        shift = self.logits.max(1, True).values
        shift.data.clamp_(min=finfo.min, max=finfo.max)
        logits = self.logits - shift
        d = logits.logsumexp(0) - math.log(2)
        for _ in range(self.bp_iters):
            s = (logits - d).logsumexp(-1, True)
            d = (logits - s).logsumexp(0) - math.log(2)
        b = (logits - (d + s)).exp()

        def log(x):
            return x.clamp(min=finfo.tiny).log()

        # Evaluate the Bethe free energy, adapting [4] Eqn 4 to one-two
        # matchings.
        b_ = (1 - b).clamp(min=0)
        internal_energy = -(b * logits.clamp(min=-1 / finfo.eps)).sum()
        # Let h be the entropy of matching each destin to one source.
        z = b / 2
        h = -(z * log(z)).sum(0)
        # Then h2 is the entropy of the distribution mapping each destin to an
        # unordered pair of sources, equal to the log of the binomial
        # coefficient: perplexity * (perplexity - 1) / 2.
        h2 = h + log(h.expm1()) - math.log(2)
        free_energy = internal_energy - h2.sum() - (b_ * log(b_)).sum()
        log_Z = shift.sum() - free_energy
        assert torch.isfinite(log_Z)
        return log_Z

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
            return torch.stack(
                [self.sample(sample_shape[1:]) for _ in range(sample_shape[0])]
            )
        # TODO initialize via .mode(), then perform a small number of MCMC steps
        # https://www.cc.gatech.edu/~vigoda/Permanent.pdf
        # https://papers.nips.cc/paper/2012/file/4c27cea8526af8cfee3be5e183ac9605-Paper.pdf
        raise NotImplementedError

    def mode(self):
        """
        Computes a maximum probability matching.
        """
        return maximum_weight_matching(self.logits)


def enumerate_one_two_matchings(num_destins):
    if num_destins == 1:
        return torch.tensor([[0, 0]])

    num_sources = num_destins * 2
    subproblem = enumerate_one_two_matchings(num_destins - 1)
    subsize = subproblem.size(0)
    result = torch.empty(
        subsize * num_sources * (num_sources - 1) // 2, num_sources, dtype=torch.long
    )

    # Iterate over pairs of sources s0<s1 matching the last destination d.
    d = num_destins - 1
    pos = 0
    for s1 in range(num_sources):
        for s0 in range(s1):
            block = result[pos : pos + subsize]
            block[:, :s0] = subproblem[:, :s0]
            block[:, s0] = d
            block[:, s0 + 1 : s1] = subproblem[:, s0 : s1 - 1]
            block[:, s1] = d
            block[:, s1 + 1 :] = subproblem[:, s1 - 1 :]
            pos += subsize
    return result


@torch.no_grad()
def maximum_weight_matching(logits):
    from scipy.optimize import linear_sum_assignment

    cost = -logits.cpu()
    cost = torch.cat([cost, cost], dim=-1)  # Duplicate destinations.
    value = linear_sum_assignment(cost.numpy())[1]
    value = torch.tensor(value, dtype=torch.long, device=logits.device)
    value %= logits.size(1)
    return value
