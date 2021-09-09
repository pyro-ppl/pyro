# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Callable

import torch
from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.inspect import get_dependencies
from pyro.nn import PyroModule
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample

from .guides import AutoGuide
from .initialization import InitMessenger, init_to_feasible
from .utils import deep_getattr, deep_setattr, helpful_support_errors


class _AutoWrapMeta(type(PyroModule), ABCMeta):
    pass


class AutoWrap(AutoGuide, metaclass=_AutoWrapMeta):
    """
    EXPERIMENTAL Abstract base class for auto guides that wrap models.

    These guides are useful in that they have access to pyro.deterministic
    sites computed during model execution. Subclasses should implement the
    :meth:`_sample` method and typically override the :meth:`_setup_prototype`
    method to perform additional setup.
    """

    def __init__(
        self,
        model: Callable,
        *,
        init_loc_fn: Callable = init_to_feasible,
    ):
        self._original_model = (model,)
        model = InitMessenger(init_loc_fn)(model)
        super().__init__(model)

    def _setup_prototype(self, *args, **kwargs) -> None:
        super()._setup_prototype(*args, **kwargs)

        # Set up dependencies.
        model = self._original_model[0]
        meta = poutine.block(get_dependencies)(
            model,
            args,
            kwargs,
            include_obs=True,
            order="prior",
        )
        self.dependencies = meta["posterior_dependencies"]
        self.upstream = defaultdict(list)
        self.downstream = defaultdict(list)
        for d, upstreams in self.dependencies.items():
            for u in upstreams:
                self.upstream[d].append(u)
                self.downstream[u].append(d)
            assert self.upstream[d]
            assert self.upstream[d][-1] == d

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        # Initialize top-level latent variables.
        self._args_kwargs = args, kwargs
        self._pending = {k: len(v) for k, v in self.upstream.items()}
        self._values = {}
        for name, count in list(self._pending.items()):
            if count == 1:
                self._values[name] = self._sample(name)

        # Run a wrapped model, intercepting sites along the way.
        with AutoWrapMessenger(self):
            self.model(*args, **kwargs)

        # Clean up.
        assert all(v == 0 for v in self._pending.values())
        values = self._values
        del self._args_kwargs
        del self._pending
        del self._values
        return values

    def _get_value(self, name: str) -> torch.Tensor:
        return self._values[name]

    def _set_value(self, name: str, value: torch.Tensor) -> None:
        self._values[name] = value
        for d in self.downstream[name]:
            self._pending[d] -= 1
            if self._pending[d] == 1:
                self._values[d] = self._sample(d)

    @abstractmethod
    def _sample(self, name: str) -> torch.Tensor:
        """
        :param str name: The name of a latent variable in the model.
        :returns: A sampled value for the latent variable specified by ``name``.
        :rtype: torch.Tensor
        """
        raise NotImplementedError


# Internal helper for AutoWrap.
class AutoWrapMessenger(Messenger):
    def __init__(self, guide: AutoWrap):
        super().__init__()
        self.guide = guide

    def _pyro_sample(self, msg: dict) -> None:
        if msg["infer"].get("is_auxiliary") or site_is_subsample(msg):
            return
        if msg["is_observed"]:
            return

        # Replay, promoting latent sites to deterministics.
        msg["is_observed"] = True
        msg["value"] = self.guide._get_value(msg["name"])
        msg["done"] = True

    def _pyro_post_sample(self, msg: dict) -> None:
        if msg["infer"].get("is_auxiliary") or site_is_subsample(msg):
            return
        self.guide._set_value(msg["name"], msg["value"])


class AutoWrapFull(AutoWrap):
    """
    EXPERIMENTAL
    """

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._unconstrained_shapes = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect the shapes of unconstrained values.
            # These may differ from the shapes of constrained values.
            with helpful_support_errors(site):
                self._unconstrained_shapes[name] = (
                    biject_to(site["fn"].support).inv(site["value"]).shape
                )

        self.locs = PyroModule()
        self.weights = PyroModule()
        self.einsums = {}
        for d, upstreams in self.upstream.items():
            d_site = self.prototype_trace.nodes[d]["fn"]
            init_loc = d_site["value"].new_zeros(self._unconstrained_shapes[d])
            deep_setattr(self.locs, torch.nn.Parameter(init_loc))
            weights = PyroModule()
            deep_setattr(self.weights, d, weights)
            for u, plates in self.dependencies[d]:
                deep_setattr(weights, u, "TODO")

    def _set_value(self, name: str, value: torch.Tensor) -> None:
        if name not in self._aux_values:
            self._aux_values[name] = "TODO"
        super()._set_value(name, value)

    def forward(self, *args, **kwargs):
        self._aux_values = {}
        values = super().forward(*args, **kwargs)
        del self._aux_values
        return values

    def _sample(self, name: str) -> torch.Tensor:
        site = self.prototype_trace.nodes[name]

        # Draw a parameter-free auxiliary variable.
        loc = deep_getattr(self.locs, name)
        zero = loc.new_zeros(()).expand_as(loc)
        aux_value = pyro.sample(
            name + "_aux",
            dist.Normal(zero, 1).to_event(1),
            infer={"is_auxiliary": True},
        ).reshape(-1)

        # Apply skew transforms, preserving volume.
        weights = deep_getattr(self.weights, name)
        for u in self.upstreams[name][:-1]:
            weight = deep_getattr(weights, u)
            aux_value = aux_value + self._aux_values[u] @ weight  # FIXME einsum
        weight = deep_getattr(weights, name)
        aux_value = aux_value + aux_value @ weight.triu(diagonal=1)  # FIXME einsum
        aux_value = aux_value + aux_value @ weight.tril(diagonal=-1)  # FIXME einsum
        self._aux_values[name] = aux_value

        # Shift and scale.
        log_scale = weight.diagonal(dim=-2, dim2=-1)  # FIXME einsum
        if poutine.get_mask() is not False:
            pyro.factor(name + "_aux_density", -log_scale.sum(-1))
        scale = log_scale.exp()
        flat_value = (aux_value + loc) * scale
        unconstrained_value = flat_value.reshape(self._unconstrained_shapes[name])
        value = biject_to(site["fn"].support)(unconstrained_value)
        return value
