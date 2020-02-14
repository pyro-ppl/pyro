# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from collections import OrderedDict

from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.trace_messenger import TraceMessenger

from pyro.contrib.funsor import to_funsor, to_data
from pyro.contrib.funsor.named_messenger import GlobalNameMessenger, NamedMessenger


class MarkovMessenger(NamedMessenger):
    pass


class IndepMessenger(GlobalNameMessenger):
    """
    Sketch of vectorized plate implementation using to_data instead of _DIM_ALLOCATOR
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
        if self.dim is not None:
            indices = to_data(self._indices, name_to_dim=OrderedDict([(self.name, self.dim)]))
        else:
            indices = to_data(self._indices)  # TODO indicate that this dim is user-visible
        self.dim, self.indices = -indices.dim(), indices.squeeze()
        return self

    def _pyro_sample(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

    def _pyro_param(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]


class EnumMessenger(GlobalNameMessenger):

    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        super().__init__()

    def _pyro_sample(self, msg):

        import funsor

        if msg["done"] or msg["is_observed"] or msg.get("expand", False) or \
                msg["infer"].get("enumerate") != "parallel":
            return

        msg["infer"]["funsor_fn"] = to_funsor(msg["fn"])
        size = msg["infer"]["funsor_fn"].inputs["value"].dtype
        msg["infer"]["funsor_value"] = funsor.Tensor(
            torch.arange(size),
            OrderedDict([(msg["name"], size)]),
            funsor.bint(size)
        )

        msg["value"] = to_data(msg["infer"]["funsor_value"])
        msg["done"] = True


class FunsorTraceMessenger(TraceMessenger):

    def _pyro_post_sample(self, msg):
        if "funsor_fn" not in msg["infer"]:
            msg["infer"]["funsor_fn"] = to_funsor(msg["fn"])
        if "funsor_value" not in msg["infer"]:
            msg["infer"]["funsor_value"] = to_funsor(msg["value"])
        return super()._pyro_post_sample(msg)
