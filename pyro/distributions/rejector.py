from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Distribution)
class ImplicitRejector(Distribution):
    """
    Rejection sampled distribution given an acceptance rate function.

    :param Distribution proposer: A distribution object that samples
        batched propsals via `proposer.sample()`.
    :param callable acceptor: A callable that inputs a batch of proposals
        and returns a batch of acceptance probabilities.
    """
    reparameterized = True

    def __init__(self, proposer, acceptor):
        self.proposer = proposer
        self.acceptor = acceptor

    def sample(self):
        # Implements parallel batched accept-reject sampling.
        sample = None
        done = None
        while True:
            proposal = self.proposer.sample()
            prob_accept = self.acceptor(proposal)
            accept = torch.bernoulli(prob_accept).byte()
            if sample is None:
                # Initialize on first iteration.
                sample = proposal
                done = accept
            elif accept.any():
                sample[accept] = proposal[accept]
                done |= accept
            if done.all():
                break
        return sample.detach()

    def batch_log_pdf(self, x):
        return self.proposal.batch_log_pdf(x) + self.acceptor(x)

    def score_function_term(self, x):
        # TODO look into caching strategies
        return self.acceptor(x)


@copy_docs_from(Distribution)
class ExplicitRejector(Distribution):
    """
    Rejection sampled distribution given a target distribution.

    :param Distribution proposer: A distribution that implements `.sample()`
        and `.batch_log_pdf()`.
    :param Distribution target: A distribution that implements
        `.batch_log_pdf()`.
    :param Variable log_scale: A batch of log scale factors satisfying:
        `proposer.batch_log_pdf(x) + log_scale >= target.batch_log_pdf(x)`.
    """
    reparameterized = True

    def __init__(self, proposer, target, log_scale):
        self.proposer = proposer
        self.target = target
        self.log_scale = log_scale

    def sample(self):
        # Implements parallel batched accept-reject sampling.
        sample = None
        done = None
        while True:
            proposal = self.proposer.sample()
            log_prob_accept = (self.target.batch_log_pdf(proposal) - self.log_scale -
                               self.proposer.batch_log_pdf(proposal))
            assert (log_prob_accept <= 0).all(), 'bad log_scale'
            accept = torch.bernoulli(torch.exp(log_prob_accept)).byte()
            if sample is None:
                # Initialize on first iteration.
                sample = proposal
                done = accept
            elif accept.any():
                sample[accept] = proposal[accept]
                done |= accept
            if done.all():
                break
        return sample

    def batch_log_pdf(self, x):
        return self.target.batch_log_pdf(x)

    def score_function_term(self, x):
        # TODO look into caching strategies
        return (self.target.batch_log_pdf(x) - self.log_scale -
                self.proposer.batch_log_pdf(x))
