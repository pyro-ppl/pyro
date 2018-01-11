from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Distribution)
class ImplicitRejector(Distribution):
    """
    Rejection sampled distribution given an acceptance rate function.

    :param Distribution propose: A proposal distribution that samples batched
        propsals via `propose()`.
    :param callable log_prob_accept: A callable that inputs a batch of proposals
        and returns a batch of log acceptance probabilities.
    """
    stateful = True
    reparameterized = True

    def __init__(self, propose, log_prob_accept):
        self.propose = propose
        self.log_prob_accept = log_prob_accept
        self._log_prob_accept_cache = None, None
        self._propose_batch_log_pdf_cache = None, None

    def _log_prob_accept(self, x):
        if x is not self._log_prob_accept_cache[0]:
            self._log_prob_accept_cache = x, self.log_prob_accept(x)
        return self._log_prob_accept_cache[1]

    def _propose_batch_log_pdf(self, x):
        if x is not self._propose_batch_log_pdf_cache[0]:
            self._propose_batch_log_pdf_cache = x, self.propose.batch_log_pdf(x)
        return self._propose_batch_log_pdf_cache[1]

    def sample(self):
        # Implements parallel batched accept-reject sampling.
        x = self.propose()
        log_prob_accept = self.log_prob_accept(x)
        # assert (log_prob_accept <= 1e-6).all(), 'bad log_scale'
        probs = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        done = torch.bernoulli(probs).byte()
        while not done.all():
            proposed_x = self.propose()
            log_prob_accept = self.log_prob_accept(proposed_x)
            prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
            accept = torch.bernoulli(prob_accept).byte() & ~done
            if accept.any():
                x[accept] = proposed_x[accept]
                done |= accept
        return x

    def batch_log_pdf(self, x):
        return self._propose_batch_log_pdf(x) + self._log_prob_accept(x)

    def score_parts(self, x):
        score_function = self._log_prob_accept(x)
        log_pdf = self.batch_log_pdf(x)
        return ScoreParts(log_pdf, score_function, log_pdf)


@copy_docs_from(Distribution)
class ExplicitRejector(ImplicitRejector):
    """
    Rejection sampled distribution given a target distribution.

    :param Distribution propose: A proposal distribution that samples batched
        proposals via `propose()`.
    :param Distribution target: A target distribution that implements
        `.batch_log_pdf()`.
    :param Variable log_scale: A batch of log scale factors satisfying:
        `propose.batch_log_pdf(x) + log_scale >= target.batch_log_pdf(x)`.
    """
    reparameterized = True

    def __init__(self, propose, target, log_scale):
        super(ExplicitRejector, self).__init__(propose, self.log_prob_accept)
        self.target = target
        self.log_scale = log_scale
        self._target_batch_log_pdf_cache = None, None

    def log_prob_accept(self, x):
        return (self._target_batch_log_pdf(x) - self.log_scale -
                self._propose_batch_log_pdf(x))

    def _target_batch_log_pdf(self, x):
        if x is not self._target_batch_log_pdf_cache[0]:
            self._target_batch_log_pdf_cache = x, self.target.batch_log_pdf(x)
        return self._target_batch_log_pdf_cache[1]

    def batch_log_pdf(self, x):
        return self._target_batch_log_pdf(x)
