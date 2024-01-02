# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC
from typing import Callable, Optional

import torch
from typing_extensions import TypedDict


class ReparamMessage(TypedDict):
    name: str
    fn: Callable
    value: Optional[torch.Tensor]
    is_observed: Optional[bool]


class ReparamResult(TypedDict):
    fn: Callable
    value: Optional[torch.Tensor]
    is_observed: bool


class Reparam(ABC):
    """
    Abstract base class for reparameterizers.

    Derived classes should implement :meth:`apply`.
    """

    # @abstractmethod  # Not abstract, for backwards compatibility.
    def apply(self, msg: ReparamMessage) -> ReparamResult:
        """
        Abstract method to apply reparameterizer.

        :param dict name: A simplified Pyro message with fields:
            - ``name: str`` the sample site's name
            - ``fn: Callable`` a distribution
            - ``value: Optional[torch.Tensor]`` an observed or initial value
            - ``is_observed: bool`` whether ``value`` is an observation
        :returns: A simplified Pyro message with fields ``fn``, ``value``, and
            ``is_observed``.
        :rtype: dict
        """

        # This default is provided for backwards compatibility only.
        # New subclasses should define .apply() and omit .__call__().
        warnings.warn(
            "Reparam.__call__() is deprecated in favor of .apply(); "
            "new subclasses should implement .apply().",
            DeprecationWarning,
        )
        new_fn, value = self(msg["name"], msg["fn"], msg["value"])
        is_observed = msg["value"] is None and value is not None
        return {"fn": new_fn, "value": value, "is_observed": is_observed}

    def __call__(self, name, fn, obs):
        """
        DEPRECATED.
        Subclasses should implement :meth:`apply` instead.
        This will be removed in a future release.
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
