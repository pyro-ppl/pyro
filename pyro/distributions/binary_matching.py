# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import torch
from torch.distributions import constraints
from torch.distributions.util import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution


def _log_factorial(n):
    if isinstance(n, torch.Tensor):
        return torch.lgamma(n + 1)
    return math.lgamma(n + 1)


def _log_binomial_coefficient(n, k):
    if k == 1:
        return n.log()
    if k == 2:
        return (n * (n - 1) / 2).log()
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


class BinaryMatchingConstraint(constraints.Constraint):
    def __init__(self, num_sources, num_destins):
        self.num_sources = num_sources
        self.num_destins = num_destins

    def check(self, value):
        if value.dim() == 0:
            warnings.warn("Invalid event_shape: ()")
            return torch.tensor(False)
        batch_shape, event_shape = value.shape[:-1], value.shape[-1:]
        if event_shape != (self.num_sources,):
            warnings.warn("Invalid event_shape: {}".format(event_shape))
            return torch.tensor(False)

        # 1. Each sources matches a single destination.
        if value.min() < 0 or value.max() >= self.num_destins:
            warnings.warn("Value out of bounds")
            return torch.tensor(False)

        counts = torch.zeros(batch_shape + (self.num_destins,))
        counts.scatter_add_(-1, value, torch.ones(value.shape))
        internal_counts, root_count = counts[..., :-1], counts[..., -1]

        # 2. Internal nodes should be either inactive or binary.
        if not ((internal_counts == 0) | (internal_counts == 2)).all():
            warnings.warn("Matching is not optionally-binary")
            return torch.tensor(False)

        # 3. Inactive sources must match the root destination.
        root = self.num_destins - 1
        if ((internal_counts == 0) & (value[..., :-1] != root)).any():
            warnings.warn("Inactive node does not match root")
            return torch.tensor(False)

        # 4. The root should match one binary node plus all inactive nodes.
        I = self.num_destins - 1
        L = self.num_sources - I
        if not (root_count == I - L + 2).all():
            warnings.warn("Root flux is incorrect")
            return torch.tensor(False)

        return torch.tensor(True)


class BinaryMatching(TorchDistribution):
    r"""
    Random matching from ``I+L`` sources to ``I+1`` destinations where:

    1.  Each source matches exactly one destination;
    2.  Each of the first ``I`` destinations matches either zero or two
        sources. We say the destination is "inactive" if it matches zero soures
        and "active" if it matches two.
    3.  Inactive sources must match the root destination.
    4.  The final "root" destination matches exactly ``I-L+2`` sources.

    Samples are represented as long tensors of shape ``(I+L,)`` taking values
    in ``{0,...,I}`` and satisfying the above constraints when treated as the
    source-to-destin mapping. The log probability of a sample ``v`` is the sum
    of edge logits, up to the log partition function ``log Z``:

    .. math::

        \log p(v) = \sum_s \text{logits}[s, v[s]] - \log Z

    Exact computations are expensive. To enable tractable approximations, set a
    number of belief propagation iterations via the ``bp_iters`` argument. The
    :meth:`log_partition_function` and :meth:`log_prob` methods use a Bethe
    free energy approximation.

    :param Tensor logits: An ``(I+L, I+1)``-shaped tensor of edge logits.
    """
    arg_constraints = {"logits": constraints.real}
    has_enumerate_support = True

    def __init__(self, logits, *, validate_args=None):
        if logits.dim() != 2:
            raise NotImplementedError("BinaryMatching does not support batching")
        self.num_sources, self.num_destins = logits.shape
        I = self.num_destins - 1
        L = self.num_sources - I
        assert I >= L - 1
        batch_shape = ()
        event_shape = (self.num_sources,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def log_partition_function(self):
        if self.bp_iters is None:
            # Brute force.
            d = self.enumerate_support()
            s = torch.arange(d.size(-1), dtype=d.dtype, device=d.device)
            return self.logits[s, d].sum(-1).logsumexp(-1)

        # Approximate mean field beliefs b via Sinkhorn iteration.
        finfo = torch.finfo(self.logits.dtype)
        shift = self.logits.data.max(1, True).values
        shift.clamp_(min=finfo.min, max=finfo.max)
        p = (self.logits - shift).exp().clamp(min=finfo.tiny ** 0.5)
        e = "TODO"
        d = 2 / p.sum(0)
        for _ in range(self.bp_iters):
            s = 1 / (p @ d)
            d = 2 / (s @ p)
        b = s[:, None] * d * p

        # Evaluate the Bethe free energy.
        b = b.clamp(min=finfo.tiny ** 0.5)
        b_ = (1 - b).clamp(min=finfo.tiny ** 0.5)
        internal_energy = -shift.sum() - (b * p.log()).sum()
        # Compute the perplexity of matching each destin to one source.
        z = b / e
        perplexity = (z * z.log()).sum(0).neg().exp()
        # Then H2 is the entropy of the distribution matching each destin to an
        # unordered pair of sources.
        H2 = _log_binomial_coefficient(perplexity[:-1], 2)
        # And HR is the entropy of the distribution matching the root to an
        # unordered set of 1+num_inactive sources.
        HR = _log_binomial_coefficient(perplexity[-1], 1 + self.num_inactive)
        raise NotImplementedError("TODO")

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        d = value
        s = torch.arange(d.size(-1), dtype=d.dtype, device=d.device)
        return self.logits[s, d].sum(-1) - self.log_partition_function
