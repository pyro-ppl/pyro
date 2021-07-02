# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.distributions import constraints
from pyro.distributions.hmm import _sequential_logmatmulexp
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class MissingDataDiscreteHMM(TorchDistribution):
    """
    HMM with discrete latent states and discrete observations, allowing for
    missing data or variable length sequences. Observations are assumed
    to be one hot encoded; rows with all zeros indicate missing data.

    .. warning:: Unlike in pyro's pyro.distributions.DiscreteHMM, which
        computes the probability of the first state as
        initial.T @ transition @ emission
        this distribution uses the standard HMM convention,
        initial.T @ emission

    :param ~torch.Tensor initial_logits: A logits tensor for an initial
        categorical distribution over latent states. Should have rightmost
        size ``state_dim`` and be broadcastable to
        ``(batch_size, state_dim)``.
    :param ~torch.Tensor transition_logits: A logits tensor for transition
        conditional distributions between latent states. Should have rightmost
        shape ``(state_dim, state_dim)`` (old, new), and be broadcastable
        to ``(batch_size, state_dim, state_dim)``.
    :param ~torch.Tensor observation_logits: A logits tensor for observation
        distributions from latent states. Should have rightmost shape
        ``(state_dim, categorical_size)``, where ``categorical_size`` is the
        dimension of the categorical output, and be broadcastable
        to ``(batch_size, state_dim, categorical_size)``.
    """

    arg_constraints = {
        "initial_logits": constraints.real_vector,
        "transition_logits": constraints.independent(constraints.real, 2),
        "observation_logits": constraints.independent(constraints.real, 2),
    }
    support = constraints.independent(constraints.nonnegative_integer, 2)

    def __init__(
        self, initial_logits, transition_logits, observation_logits, validate_args=None
    ):
        if initial_logits.dim() < 1:
            raise ValueError(
                "expected initial_logits to have at least one dim, "
                "actual shape = {}".format(initial_logits.shape)
            )
        if transition_logits.dim() < 2:
            raise ValueError(
                "expected transition_logits to have at least two dims, "
                "actual shape = {}".format(transition_logits.shape)
            )
        if observation_logits.dim() < 2:
            raise ValueError(
                "expected observation_logits to have at least two dims, "
                "actual shape = {}".format(transition_logits.shape)
            )
        shape = broadcast_shape(
            initial_logits.shape[:-1],
            transition_logits.shape[:-2],
            observation_logits.shape[:-2],
        )
        if len(shape) == 0:
            shape = torch.Size([1])
        batch_shape = shape
        event_shape = (1, observation_logits.shape[-1])
        self.initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        self.transition_logits = transition_logits - transition_logits.logsumexp(
            -1, True
        )
        self.observation_logits = observation_logits - observation_logits.logsumexp(
            -1, True
        )
        super(MissingDataDiscreteHMM, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def log_prob(self, value):
        """
        :param ~torch.Tensor value: One-hot encoded observation. Must be
            real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
            Missing data is represented by zeros, i.e.
            ``value[batch, step, :] == tensor([0, ..., 0])``.
            Variable length observation sequences can be handled by padding
            the sequence with zeros at the end.
        """

        assert value.shape[-1] == self.event_shape[1]

        # Combine observation and transition factors.
        value_logits = torch.matmul(
            value, torch.transpose(self.observation_logits, -2, -1)
        )
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]

        # Eliminate time dimension.
        result = _sequential_logmatmulexp(result)

        # Combine initial factor.
        result = self.initial_logits + value_logits[..., 0, :] + result.logsumexp(-1)

        # Marginalize out final state.
        result = result.logsumexp(-1)
        return result
