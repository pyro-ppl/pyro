# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch


class Reparam(ABC):
    """
    Base class for reparameterizers.
    """
    @abstractmethod
    def __call__(self, name, fn, obs):
        """
        :param str name: A sample site name.
        :param ~pyro.distributions.TorchDistribution fn: A distribution.
        :param ~torch.Tensor obs: Observed value or None.
        :return: A pair (``new_fn``, ``value``).
        """
        return fn, obs

    def _unwrap(self, fn):
        """
        Unwrap Independent distributions.
        """
        event_dim = fn.event_dim
        while isinstance(fn, torch.distributions.Independent):
            fn = fn.base_dist
        return fn, event_dim

    def _wrap(self, fn, event_dim):
        """
        Wrap in Independent distributions.
        """
        if fn.event_dim < event_dim:
            fn = fn.to_event(event_dim - fn.event_dim)
        assert fn.event_dim == event_dim
        return fn
