# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class Unit(TorchDistribution):
    """
    Trivial nonnormalized distribution representing the unit type.

    The unit type has a single value with no data, i.e. ``value.numel() == 0``.

    This is used for :func:`pyro.factor` statements.
    """
    arg_constraints = {'log_factor': constraints.real}
    support = constraints.real

    def __init__(self, log_factor, validate_args=None):
        log_factor = torch.as_tensor(log_factor)
        batch_shape = log_factor.shape
        event_shape = torch.Size((0,))  # This satisfies .numel() == 0.
        self.log_factor = log_factor
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Unit, _instance)
        new.log_factor = self.log_factor.expand(batch_shape)
        super(Unit, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        return self.log_factor.new_empty(sample_shape + self.shape())

    def log_prob(self, value):
        shape = broadcast_shape(self.batch_shape, value.shape[:-1])
        return self.log_factor.expand(shape)
