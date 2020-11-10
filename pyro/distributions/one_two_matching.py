# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from .torch_distribution import TorchDistribution


class OneTwoMatchingConstraint(constraints.Constraint):
    def __init__(self, num_destins):
        self.num_destins = num_destins
        self.num_sources = 2 * num_destins

    def check(self, value):
        if value.dim() == 0:
            warnings.warn("Invalid event_shape: ()")
            return False
        batch_shape, event_shape = value.shape[:-1], value.shape[-1:]
        if event_shape != (self.num_sources,):
            warnings.warn("Invalid event_shape: {}".format(event_shape))
            return False
        if value.min() < 0 or value.max() >= self.num_destins:
            warnings.warn("Value out of bounds")
            return False
        counts = torch.zeros(batch_shape + (self.num_destins,))
        counts.scatter_add_(-1, value, torch.ones(value.shape))
        if (counts != 2).any():
            warnings.warn("Matching is not binary")
            return False
        return True


class OneTwoMatching(TorchDistribution):
    """
    Random matching from ``2 N`` sources to ``N`` destinations where each
    source matches one destination and each destination matches two sources.

    The :meth:`log_partition_function` and :meth:`log_prob` methods use a Bethe
    approximation [1,2]. This currently does not implement :meth:`sample`.

    **References:**

    [1] Bert Huang, Tony Jebara (2009)
        "Approximating the Permanent with Belief Propagation"
        https://arxiv.org/pdf/0908.1769.pdf
    [2] Pascal O. Vontobel (2012)
        "The Bethe Permanent of a Non-Negative Matrix"
        https://arxiv.org/pdf/1107.4196.pdf

    :param Tensor logits: An ``(2 * N, N)``-shaped tensor of edge logits.
    """
    arg_constraints = {"logits": constraints.real}
    has_enumerate_support = True

    def __init__(self, logits, *, validate_args=None):
        if logits.dim() != 2:
            raise NotImplementedError("OneTwoMatching does not support batching")
        self.num_sources, self.num_destins = logits.shape
        assert self.num_sources == 2 * self.num_destins
        self.logits = logits
        batch_shape = ()
        event_shape = (self.num_sources,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return OneTwoMatchingConstraint(self.num_destins)

    @lazy_property
    def log_partition_function(self):
        return log_count_two_matchings(self.logits)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        i = value
        j = torch.arange(i.size(-1), dtype=i.dtype, device=i.device)
        return self.logits[i, j].sum(-1) - self.log_partition_function

    def enumerate_support(self, expand=True):
        return enumerate_one_two_matchings(self.num_destins)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape:
            return torch.stack([self.sample(sample_shape[1:])
                                for _ in range(sample_shape[0])])
        # Consider initialzing with heuristic or MAP
        # https://arxiv.org/pdf/0709.1190
        # http://proceedings.mlr.press/v15/huang11a/huang11a.pdf
        # followed by a small number of MCMC steps
        # http://www.cs.toronto.edu/~mvolkovs/nips2012_sampling.pdf
        raise NotImplementedError


def log_count_two_matchings(logits):
    raise NotImplementedError("TODO")


def enumerate_one_two_matchings(num_destins):
    if num_destins == 1:
        return torch.tensor([[0, 0]])

    num_sources = num_destins * 2
    subproblem = enumerate_one_two_matchings(num_destins - 1)
    subsize = subproblem.size(0)
    result = torch.empty(subsize * num_sources * (num_sources - 1) // 2,
                         num_sources,
                         dtype=torch.long)

    # Iterate over pairs of sources (s0<s1) chosen by the last destination d.
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
