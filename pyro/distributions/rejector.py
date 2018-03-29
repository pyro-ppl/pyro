from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.torch_distribution import TorchDistribution


class Rejector(TorchDistribution):
    """
    Rejection sampled distribution given an acceptance rate function.

    :param Distribution propose: A proposal distribution that samples batched
        proposals via ``propose()``. :meth:`rsample` supports a ``sample_shape``
        arg only if ``propose()`` supports a ``sample_shape`` arg.
    :param callable log_prob_accept: A callable that inputs a batch of
        proposals and returns a batch of log acceptance probabilities.
    :param log_scale: Total log probability of acceptance.
    """
    has_rsample = True

    def __init__(self, propose, log_prob_accept, log_scale):
        self.propose = propose
        self.log_prob_accept = log_prob_accept
        self._log_scale = log_scale

        # These LRU(1) caches allow work to be shared across different method calls.
        self._log_prob_accept_cache = None, None
        self._propose_log_prob_cache = None, None

    def _log_prob_accept(self, x):
        if x is not self._log_prob_accept_cache[0]:
            self._log_prob_accept_cache = x, self.log_prob_accept(x) - self._log_scale
        return self._log_prob_accept_cache[1]

    def _propose_log_prob(self, x):
        if x is not self._propose_log_prob_cache[0]:
            self._propose_log_prob_cache = x, self.propose.log_prob(x)
        return self._propose_log_prob_cache[1]

    def rsample(self, sample_shape=torch.Size()):
        # Implements parallel batched accept-reject sampling.
        x = self.propose(sample_shape) if sample_shape else self.propose()
        log_prob_accept = self.log_prob_accept(x)
        probs = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        done = torch.bernoulli(probs).byte()
        while not done.all():
            proposed_x = self.propose(sample_shape) if sample_shape else self.propose()
            log_prob_accept = self.log_prob_accept(proposed_x)
            prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
            accept = torch.bernoulli(prob_accept).byte() & ~done
            if accept.any():
                x[accept] = proposed_x[accept]
                done |= accept
        return x

    def log_prob(self, x):
        return self._propose_log_prob(x) + self._log_prob_accept(x)

    def score_parts(self, x):
        score_function = self._log_prob_accept(x)
        log_prob = self.log_prob(x)
        return ScoreParts(log_prob, score_function, log_prob)
