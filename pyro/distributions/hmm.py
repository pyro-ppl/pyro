from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.util import ignore_jit_warnings


def _logmatmulexp(x, y):
    """
    Numerically stable version of ``(x.log() @ y.log()).exp()``.
    """
    x_shift = x.max(-1, keepdim=True)[0]
    y_shift = y.max(-2, keepdim=True)[0]
    xy = torch.matmul((x - x_shift).exp(), (y - y_shift).exp()).log()
    return xy + x_shift + y_shift


@ignore_jit_warnings()
def _sequential_logmatmulexp(logits):
    """
    For a tensor ``x`` whose time dimension is -3, computes::

        x[..., 0, :, :] @ x[..., 1, :, :] @ ... @ x[..., T-1, :, :]

    but does so numerically stably in log space.
    """
    batch_shape = logits.shape[:-3]
    state_dim = logits.size(-1)
    while logits.size(-3) > 1:
        time = logits.size(-3)
        even_time = time // 2 * 2
        even_part = logits[..., :even_time, :, :]
        x_y = even_part.reshape(batch_shape + (even_time // 2, 2, state_dim, state_dim))
        x, y = x_y.unbind(-3)
        contracted = _logmatmulexp(x, y)
        if time > even_time:
            contracted = torch.cat((contracted, logits[..., -1:, :, :]), dim=-3)
        logits = contracted
    return logits.squeeze(-3)


class DiscreteHMM(TorchDistribution):
    """
    Hidden Markov Model with discrete latent state and arbitrary observation
    distribution. This uses [1] to parallelize over time, achieving
    O(log(time)) parallel complexity.

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    :param torch.Tensor initial_logits: A logits tensor for an initial
        categorical distribution over latent states. Should have rightmost size
        ``state_dim`` and be broadcastable to ``batch_shape + (state_dim,)``.
    :param torch.Tensor transition_logits: A logits tensor for transition
        conditional distributions between latent states. Should have rightmost
        shape ``(num_steps, state_dim, state_dim)`` (time, old, new), and be
        broadcastable to ``batch_shape + (num_steps, state_dim, state_dim)``.
    :param torch.distriburtions.Distribution observation_dist: A conditional
        distribution of observed data conditioned on latent state. The
        ``.batch_shape`` should have rightmost size ``state_dim`` and be
        broadcastable to ``batch_shape + (num_steps + state_dim)``. The
        ``.event_shape`` may be arbitrary.
    """
    arg_constraints = {"initial_logits": constraints.real,
                       "transition_logits": constraints.real}

    def __init__(self, initial_logits, transition_logits, observation_dist, validate_args=None):
        if initial_logits.dim() < 1:
            raise ValueError
        if transition_logits.dim() < 2:
            raise ValueError
        if len(observation_dist.batch_shape) < 1:
            raise ValueError
        time_shape = broadcast_shape(transition_logits.shape[-3:-2],
                                     observation_dist.batch_shape[-2:-1])
        if not time_shape:
            raise ValueError
        event_shape = time_shape + observation_dist.event_shape
        batch_shape = broadcast_shape(initial_logits.shape[:-1],
                                      transition_logits.shape[:-3],
                                      observation_dist.batch_shape[:-2])
        self.initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        self.transition_logits = transition_logits - transition_logits.logsumexp(-1, True)
        self.observation_dist = observation_dist
        super(DiscreteHMM, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        if broadcast_shape(self.batch_shape, batch_shape) == self.batch_shape:
            return self
        raise NotImplementedError

    def log_prob(self, value):
        # Combine observation and transition factors.
        value = value.unsqueeze(-1 - self.observation_dist.event_dim)
        observation_logits = self.observation_dist.log_prob(value)
        result = self.transition_logits + observation_logits.unsqueeze(-2)

        # Eliminate time dimension.
        result = _sequential_logmatmulexp(result)

        # Combine initial factor.
        result = _logmatmulexp(self.initial_logits.unsqueeze(-2), result).squeeze(-2)

        # Marginalize out final state.
        result = result.logsumexp(-1)
        return result
