# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
This file contains reimplementations of some of Pyro's core enumeration machinery,
which should eventually be drop-in replacements for the current versions.
"""
import torch
from collections import OrderedDict

from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.trace_messenger import TraceMessenger as OrigTraceMessenger

from pyro.contrib.funsor import to_funsor, to_data
from pyro.contrib.funsor.named_messenger import GlobalNameMessenger, NamedMessenger


class MarkovMessenger(NamedMessenger):
    """
    NamedMessenger is meant to be a drop-in replacement for pyro.markov.
    """
    pass


class IndepMessenger(GlobalNameMessenger):
    """
    Sketch of vectorized plate implementation using to_data instead of _DIM_ALLOCATOR.
    Should eventually be a drop-in replacement for pyro.plate.
    """
    def __init__(self, name=None, size=None, dim=None):
        assert size > 1
        assert dim is None or dim < 0
        super().__init__()
        self.name = name
        self.size = size
        self.dim = dim

        import funsor

        self._indices = funsor.Tensor(
            torch.arange(self.size, dtype=torch.long),
            OrderedDict([(self.name, funsor.bint(self.size))]),
            self.size
        )

    def __enter__(self):
        super().__enter__()  # do this first to take care of globals recycling
        name_to_dim = OrderedDict([(self.name, self.dim)]) if self.dim is not None else self.dim
        indices = to_data(self._indices, name_to_dim=name_to_dim)  # TODO indicate that this dim is user-visible
        # extract the dimension allocated by to_data to match plate's current behavior
        self.dim, self.indices = -indices.dim(), indices.squeeze()
        return self

    def _pyro_sample(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

    def _pyro_param(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]


class EnumMessenger(GlobalNameMessenger):
    """
    This version of EnumMessenger uses to_data to allocate a fresh enumeration dim
    for each discrete sample site.
    """
    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        super().__init__()

    def _pyro_sample(self, msg):

        import funsor

        if msg["done"] or msg["is_observed"] or msg.get("expand", False) or \
                msg["infer"].get("enumerate") != "parallel":
            return

        if msg["infer"].get("num_samples", None) is not None:
            raise NotImplementedError("TODO implement multiple sampling")

        msg["infer"]["funsor_fn"] = to_funsor(msg["fn"])
        size = msg["infer"]["funsor_fn"].inputs["value"].dtype
        msg["infer"]["funsor_value"] = funsor.Tensor(
            torch.arange(size),
            OrderedDict([(msg["name"], size)]),
            funsor.bint(size)
        )

        msg["value"] = to_data(msg["infer"]["funsor_value"])
        msg["done"] = True


class TraceMessenger(OrigTraceMessenger):
    """
    This version of TraceMessenger does its packing online instead of after the fact,
    converting all distributions and values to Funsors as soon as they are available.
    """
    def _pyro_post_sample(self, msg):
        if "funsor_fn" not in msg["infer"]:
            msg["infer"]["funsor_fn"] = to_funsor(msg["fn"])
        if "funsor_value" not in msg["infer"]:
            msg["infer"]["funsor_value"] = to_funsor(msg["value"])
        return super()._pyro_post_sample(msg)
