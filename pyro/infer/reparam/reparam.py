# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC
from typing import Callable, Optional, TypedDict

import torch

ReparamMessage = TypedDict(
    "ReparamMessage",
    name=str,
    fn=Callable,
    value=Optional[torch.Tensor],
    is_observded=Optional[bool],
)

ReparamResult = TypedDict(
    "ReparamResult",
    fn=Callable,
    value=Optional[torch.Tensor],
)


class Reparam(ABC):
    """
    Base class for reparameterizers.
    """

    # @abstractmethod  # Not abstract, for backwards compatibility.
    def apply(self, msg: ReparamMessage) -> ReparamResult:
        """
        Abstract method to apply reparameterizer.

        :param dict name: A simplified Pyro message with fields:
            - ``fn``
            - TODO
        """

        # This default is provided for backwards compatibility only.
        # New subclasses should define .apply() and omit .__call__().
        warnings.warn(
            "Reparam.__call__() is deprecated in favor of .apply(); "
            "new subclasses should implement .apply().",
            DeprecationWarning,
        )
        return self(msg["name"], msg["fn"], msg["value"])

    def __call__(self, name, fn, obs):
        """
        DEPRECATED. Implement :meth:`apply` instead.
        """
        raise NotImplementedError

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
