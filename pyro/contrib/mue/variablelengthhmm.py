# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.distributions import constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.hmm import _sequential_logmatmulexp
from pyro.distributions.util import broadcast_shape


class VariableLengthDiscreteHMM(TorchDistribution):
    """
    HMM with discrete latent states and discrete observations, allowing for
    variable length sequences.
    """
    arg_constraints = {"initial_logits": constraints.real,
                       "transition_logits": constraints.real,
                       "observation_logits": constraints.real}

    def __init__(self, initial_logits, transition_logits, observation_logits,
                 validate_args=None):
        if initial_logits.dim() < 1:
            raise ValueError(
                    "expected initial_logits to have at least one dim, "
                    "actual shape = {}".format(initial_logits.shape))
        if transition_logits.dim() < 2:
            raise ValueError(
                    "expected transition_logits to have at least two dims, "
                    "actual shape = {}".format(transition_logits.shape))
        if observation_logits.dim() < 2:
            raise ValueError(
                    "expected observation_logits to have at least two dims, "
                    "actual shape = {}".format(transition_logits.shape))
        shape = broadcast_shape(initial_logits.shape[:-1] + (1,),
                                transition_logits.shape[:-2],
                                observation_logits.shape[:-2])
        batch_shape = shape
        event_shape = (1, observation_logits.shape[-1])
        self.initial_logits = (initial_logits -
                               initial_logits.logsumexp(-1, True))
        self.transition_logits = (transition_logits -
                                  transition_logits.logsumexp(-1, True))
        self.observation_logits = (observation_logits -
                                   observation_logits.logsumexp(-1, True))
        super(VariableLengthDiscreteHMM, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        """Warning: unlike in pyro's DiscreteHMM, which computes the
        probability of the first state as
        initial.T @ transition @ emission
        this distribution uses the standard HMM convention,
        initial.T @ emission
        """
        # observation_logits:
        # batch_shape (option) x state_dim x observation_dim
        # value:
        # batch_shape (option) x num_steps x observation_dim
        # value_logits
        # batch_shape (option) x num_steps x state_dim (new)
        # transition_logits:
        # batch_shape (option) x state_dim (old) x state_dim (new)
        # result 1
        # batch_shape (option) x num_steps-1 x state_dim (old) x state_dim (new)
        # result 2
        # batch_shape (option) x state_dim (old) x state_dim (new)
        # initial_logits
        # batch_shape (option) x state_dim
        # result 3
        # batch_shape (option)

        # Combine observation and transition factors.
        value_logits = torch.matmul(
                value, torch.transpose(self.observation_logits, -2, -1))
        result = (self.transition_logits.unsqueeze(-3) +
                  value_logits[..., 1:, None, :])

        # Eliminate time dimension.
        result = _sequential_logmatmulexp(result)

        # Combine initial factor.
        result = (self.initial_logits + value_logits[..., 0, :]
                  + result.logsumexp(-1))

        # Marginalize out final state.
        result = result.logsumexp(-1)
        return result
