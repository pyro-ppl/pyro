# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor

from pyro.distributions.util import copy_docs_from
from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.subsample_messenger import SubsampleMessenger as OrigSubsampleMessenger

from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.contrib.funsor.handlers.enum_messenger import MarkovMessenger
from pyro.contrib.funsor.handlers.named_messenger import DimType, GlobalNamedMessenger

funsor.set_backend("torch")


class IndepMessenger(GlobalNamedMessenger):
    """
    Sketch of vectorized plate implementation using to_data instead of _DIM_ALLOCATOR.
    """
    def __init__(self, name=None, size=None, dim=None, indices=None):
        assert size > 1
        assert dim is None or dim < 0
        super().__init__()
        self.name = name
        self.size = size
        self.dim = dim
        if not hasattr(self, "_full_size"):
            self._full_size = size
        if indices is None:
            indices = funsor.ops.new_arange(funsor.tensor.get_default_prototype(), self.size)
        assert len(indices) == size

        self._indices = funsor.Tensor(
            indices, OrderedDict([(self.name, funsor.bint(self.size))]), self._full_size
        )

    def __enter__(self):
        super().__enter__()  # do this first to take care of globals recycling
        name_to_dim = OrderedDict([(self.name, self.dim)]) if self.dim is not None else OrderedDict()
        indices = to_data(self._indices, name_to_dim=name_to_dim, dim_type=DimType.VISIBLE)
        # extract the dimension allocated by to_data to match plate's current behavior
        self.dim, self.indices = -indices.dim(), indices.squeeze()
        return self

    def _pyro_sample(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

    def _pyro_param(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]


@copy_docs_from(OrigSubsampleMessenger)
class SubsampleMessenger(IndepMessenger):

    def __init__(self, name, size=None, subsample_size=None, subsample=None, dim=None,
                 use_cuda=None, device=None):
        size, subsample_size, indices = OrigSubsampleMessenger._subsample(
            name, size, subsample_size, subsample, use_cuda, device)
        self.subsample_size = subsample_size
        self._full_size = size
        self._scale = size / subsample_size
        # initialize other things last
        super().__init__(name, subsample_size, dim, indices)

    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        msg["scale"] = msg["scale"] * self._scale

    def _pyro_param(self, msg):
        super()._pyro_param(msg)
        msg["scale"] = msg["scale"] * self._scale

    def _pyro_post_param(self, msg):
        event_dim = msg["kwargs"].get("event_dim")
        if self.dim is not None and event_dim is not None and self.subsample_size < self._full_size:
            event_shape = msg["value"].shape[len(msg["value"].shape) - event_dim:]
            funsor_value = to_funsor(msg["value"], output=funsor.reals(*event_shape))
            if self.name in funsor_value.inputs:
                new_value = to_data(funsor_value(**{self.name: self._indices}))
                if hasattr(msg["value"], "_pyro_unconstrained_param"):
                    param = msg["value"]._pyro_unconstrained_param
                else:
                    param = msg["value"].unconstrained()

                if not hasattr(param, "_pyro_subsample"):
                    param._pyro_subsample = {}  # TODO is this going to persist correctly?

                param._pyro_subsample[self.dim - event_dim] = self.indices
                new_value._pyro_unconstrained_param = param
                msg["value"] = new_value

    def _pyro_post_subsample(self, msg):
        event_dim = msg["kwargs"].get("event_dim")
        if self.dim is not None and event_dim is not None and self.subsample_size < self._full_size:
            event_shape = msg["value"].shape[len(msg["value"].shape) - event_dim:]
            funsor_value = to_funsor(msg["value"], output=funsor.reals(*event_shape))
            if self.name in funsor_value.inputs:
                msg["value"] = to_data(funsor_value(**{self.name: self._indices}))


class SequentialPlateMessenger(MarkovMessenger):
    def __init__(self, name=None, size=None, dim=None):
        self.name, self.size, self.dim, self.counter = name, size, dim, 0
        super().__init__(history=0, keep=True)

    def _pyro_sample(self, msg):
        frame = CondIndepStackFrame(self.name, None, self.size, self.counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

    def _pyro_param(self, msg):
        frame = CondIndepStackFrame(self.name, None, self.size, self.counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]


class PlateMessenger(SubsampleMessenger):
    """
    Combines new IndepMessenger implementation with existing BroadcastMessenger.
    Should eventually be a drop-in replacement for pyro.plate.
    """
    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        BroadcastMessenger._pyro_sample(msg)

    def __iter__(self):
        c = SequentialPlateMessenger(self.name, self.size, self.dim)
        for i in c(range(self.size)):
            c.counter += 1
            yield i
