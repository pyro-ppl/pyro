# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import Categorical, OneHotCategorical

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

    def sample(self, sample_shape=torch.Size([])):
        """
        :param ~torch.Size sample_shape: Sample shape, last dimension must be
            ``num_steps`` and must be broadcastable to
            ``(batch_size, num_steps)``. batch_size must be int not tuple.
        """
        # shape: batch_size x num_steps x categorical_size
        shape = broadcast_shape(
            torch.Size(list(self.batch_shape) + [1, 1]),
            torch.Size(list(sample_shape) + [1]),
            torch.Size((1, 1, self.event_shape[-1])),
        )
        # state: batch_size x state_dim
        state = OneHotCategorical(logits=self.initial_logits).sample()
        # sample: batch_size x num_steps x categorical_size
        sample = torch.zeros(shape)
        for i in range(shape[-2]):
            # batch_size x 1 x state_dim @
            # batch_size x state_dim x categorical_size
            obs_logits = torch.matmul(
                state.unsqueeze(-2), self.observation_logits
            ).squeeze(-2)
            sample[:, i, :] = OneHotCategorical(logits=obs_logits).sample()
            # batch_size x 1 x state_dim @
            # batch_size x state_dim x state_dim
            trans_logits = torch.matmul(
                state.unsqueeze(-2), self.transition_logits
            ).squeeze(-2)
            state = OneHotCategorical(logits=trans_logits).sample()

        return sample

    def filter(self, value):
        """
        Compute the marginal probability of the state variable at each
        step conditional on the previous observations.

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        # batch_size x num_steps x state_dim
        shape = broadcast_shape(
            torch.Size(list(self.batch_shape) + [1, 1]),
            torch.Size(list(value.shape[:-1]) + [1]),
            torch.Size((1, 1, self.initial_logits.shape[-1])),
        )
        filter = torch.zeros(shape)

        # Combine observation and transition factors.
        # batch_size x num_steps x state_dim
        value_logits = torch.matmul(
            value, torch.transpose(self.observation_logits, -2, -1)
        )
        # batch_size x num_steps-1 x state_dim x state_dim
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]

        # Forward pass. (This could be parallelized using the
        # Sarkka & Garcia-Fernandez method.)
        filter[..., 0, :] = self.initial_logits + value_logits[..., 0, :]
        filter[..., 0, :] = filter[..., 0, :] - torch.logsumexp(
            filter[..., 0, :], -1, True
        )
        for i in range(1, shape[-2]):
            filter[..., i, :] = torch.logsumexp(
                filter[..., i - 1, :, None] + result[..., i - 1, :, :], -2
            )
            filter[..., i, :] = filter[..., i, :] - torch.logsumexp(
                filter[..., i, :], -1, True
            )
        return filter

    def smooth(self, value):
        """
        Compute posterior expected value of state at each position (smoothing).

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        # Compute filter and initialize.
        filter = self.filter(value)
        shape = filter.shape
        backfilter = torch.zeros(shape)

        # Combine observation and transition factors.
        # batch_size x num_steps x state_dim
        value_logits = torch.matmul(
            value, torch.transpose(self.observation_logits, -2, -1)
        )
        # batch_size x num_steps-1 x state_dim x state_dim
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]
        # Construct backwards filter.
        for i in range(shape[-2] - 1, 0, -1):
            backfilter[..., i - 1, :] = torch.logsumexp(
                backfilter[..., i, None, :] + result[..., i - 1, :, :], -1
            )

        # Compute smoothed version.
        smooth = filter + backfilter
        smooth = smooth - torch.logsumexp(smooth, -1, True)
        return smooth

    def sample_states(self, value):
        """
        Sample states with forward filtering-backward sampling algorithm.

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        filter = self.filter(value)
        shape = filter.shape
        joint = filter.unsqueeze(-1) + self.transition_logits.unsqueeze(-3)
        states = torch.zeros(shape[:-1], dtype=torch.long)
        states[..., -1] = Categorical(logits=filter[..., -1, :]).sample()
        for i in range(shape[-2] - 1, 0, -1):
            logits = torch.gather(
                joint[..., i - 1, :, :],
                -1,
                states[..., i, None, None]
                * torch.ones([shape[-1], 1], dtype=torch.long),
            ).squeeze(-1)
            states[..., i - 1] = Categorical(logits=logits).sample()
        return states

    def map_states(self, value):
        """
        Compute maximum a posteriori (MAP) estimate of state variable with
        Viterbi algorithm.

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        # Setup for Viterbi.
        # batch_size x num_steps x state_dim
        shape = broadcast_shape(
            torch.Size(list(self.batch_shape) + [1, 1]),
            torch.Size(list(value.shape[:-1]) + [1]),
            torch.Size((1, 1, self.initial_logits.shape[-1])),
        )
        state_logits = torch.zeros(shape)
        state_traceback = torch.zeros(shape, dtype=torch.long)

        # Combine observation and transition factors.
        # batch_size x num_steps x state_dim
        value_logits = torch.matmul(
            value, torch.transpose(self.observation_logits, -2, -1)
        )
        # batch_size x num_steps-1 x state_dim x state_dim
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]

        # Forward pass.
        state_logits[..., 0, :] = self.initial_logits + value_logits[..., 0, :]
        for i in range(1, shape[-2]):
            transit_weights = (
                state_logits[..., i - 1, :, None] + result[..., i - 1, :, :]
            )
            state_logits[..., i, :], state_traceback[..., i, :] = torch.max(
                transit_weights, -2
            )
        # Traceback.
        map_states = torch.zeros(shape[:-1], dtype=torch.long)
        map_states[..., -1] = torch.argmax(state_logits[..., -1, :], -1)
        for i in range(shape[-2] - 1, 0, -1):
            map_states[..., i - 1] = torch.gather(
                state_traceback[..., i, :], -1, map_states[..., i].unsqueeze(-1)
            ).squeeze(-1)
        return map_states

    def given_states(self, states):
        """
        Distribution conditional on the state variable.

        :param ~torch.Tensor map_states: State trajectory. Must be
            integer-valued (long) and broadcastable to
            ``(batch_size, num_steps)``.
        """
        shape = broadcast_shape(
            list(self.batch_shape) + [1, 1],
            list(states.shape[:-1]) + [1, 1],
            [1, 1, self.observation_logits.shape[-1]],
        )
        states_index = states.unsqueeze(-1) * torch.ones(shape, dtype=torch.long)
        obs_logits = self.observation_logits * torch.ones(shape)
        logits = torch.gather(obs_logits, -2, states_index)
        return OneHotCategorical(logits=logits)

    def sample_given_states(self, states):
        """
        Sample an observation conditional on the state variable.

        :param ~torch.Tensor map_states: State trajectory. Must be
            integer-valued (long) and broadcastable to
            ``(batch_size, num_steps)``.
        """
        conditional = self.given_states(states)
        return conditional.sample()
