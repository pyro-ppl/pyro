from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class DiscreteHMM(TorchDistribution):
    arg_constraints = {"transition_logits": constraints.simplex,
                       "emission_logits": constraints.simplex}

    def __init__(self, transition_logits, emission_logits, validate_args=None):
        if transition_logits.dim() < 3:
            raise ValueError
        if emission_logits.dim() < 2:
            raise ValueError
        event_shape = emission_logits.shape[-2:]
        batch_shape = broadcast_shape(transition_logits.shape[:-3], emission_logits.shape[:-2])
        self.transition_logits = transition_logits
        self.emission_logits = emission_logits
        super(DiscreteHMM, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        if self._validate_args:
            time, state_dim = event_shape
            if transition_logits.shape[-3:] != (time - 1, state_dim, state_dim):
                raise ValueError

    def log_prob(self, value):
        # Combine emission and transition factors.
        emission_part = self.emission_logits.gather(-1, value)
        result = self.transition_logits * emission_part[..., :-1, :].unsqueeze(-1)
        result[..., -1] *= emission_part[..., -1].unsqueeze(-2)

        # Perform contraction.
        batch_shape = result.shape[:-3]
        state_dim = result.size(-1)
        while result.size(-3) > 1:
            time = result.size(-3)
            even_time = time // 2 * 2
            even_part = result[..., :even_time, :, :]
            batched = even_part.reshape(batch_shape + (even_time // 2, 2, state_dim, state_dim))
            x, y = batched.unbind(-3)
            contracted = torch.matmul(x.exp(), y.exp()).log()  # TODO stabilize
            if time < even_time:
                contracted = torch.cat(contracted, result[..., -1:, :, :])
            result = contracted
        result = result.reshape(batch_shape + (state_dim * state_dim,))
        return result.logsumexp(-1)
