# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from numbers import Number

import funsor

from pyro.distributions.util import copy_docs_from
from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.messenger import Messenger
from pyro.poutine.subsample_messenger import SubsampleMessenger as OrigSubsampleMessenger
from pyro.util import ignore_jit_warnings

from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.contrib.funsor.handlers.named_messenger import DimRequest, DimType, GlobalNamedMessenger

funsor.set_backend("torch")


class IndepMessenger(GlobalNamedMessenger):
    """
    Vectorized plate implementation using to_data instead of _DIM_ALLOCATOR.
    """
    def __init__(self, name=None, size=None, dim=None, indices=None):
        assert size > 1
        assert dim is None or dim < 0
        super().__init__()
        # without a name or dim, treat as a "vectorize" effect and allocate a non-visible dim
        self.dim_type = DimType.GLOBAL if name is None and dim is None else DimType.VISIBLE
        self.name = name if name is not None else funsor.interpreter.gensym("PLATE")
        self.size = size
        self.dim = dim
        if not hasattr(self, "_full_size"):
            self._full_size = size
        if indices is None:
            indices = funsor.ops.new_arange(funsor.tensor.get_default_prototype(), self.size)
        assert len(indices) == size

        self._indices = funsor.Tensor(
            indices, OrderedDict([(self.name, funsor.Bint[self.size])]), self._full_size
        )

    def __enter__(self):
        super().__enter__()  # do this first to take care of globals recycling
        name_to_dim = OrderedDict([(self.name, DimRequest(self.dim, self.dim_type))])
        indices = to_data(self._indices, name_to_dim=name_to_dim)
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

    def __init__(self, name=None, size=None, subsample_size=None, subsample=None, dim=None,
                 use_cuda=None, device=None):
        size, subsample_size, indices = OrigSubsampleMessenger._subsample(
            name, size, subsample_size, subsample, use_cuda, device)
        self.subsample_size = subsample_size
        self._full_size = size
        self._scale = float(size) / subsample_size
        # initialize other things last
        super().__init__(name, subsample_size, dim, indices)

    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        msg["scale"] = msg["scale"] * self._scale

    def _pyro_param(self, msg):
        super()._pyro_param(msg)
        msg["scale"] = msg["scale"] * self._scale

    def _subsample_site_value(self, value, event_dim=None):
        if self.dim is not None and event_dim is not None and self.subsample_size < self._full_size:
            event_shape = value.shape[len(value.shape) - event_dim:]
            funsor_value = to_funsor(value, output=funsor.Reals[event_shape])
            if self.name in funsor_value.inputs:
                return to_data(funsor_value(**{self.name: self._indices}))
        return value

    def _pyro_post_param(self, msg):
        event_dim = msg["kwargs"].get("event_dim")
        new_value = self._subsample_site_value(msg["value"], event_dim)
        if new_value is not msg["value"]:
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
        msg["value"] = self._subsample_site_value(msg["value"], event_dim)


class PlateMessenger(SubsampleMessenger):
    """
    Combines new IndepMessenger implementation with existing BroadcastMessenger.
    Should eventually be a drop-in replacement for pyro.plate.
    """
    def __enter__(self):
        super().__enter__()
        return self.indices  # match pyro.plate behavior

    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        BroadcastMessenger._pyro_sample(msg)

    def __iter__(self):
        return iter(_SequentialPlateMessenger(self.name, self.size, self._indices.data.squeeze(), self._scale))


class _SequentialPlateMessenger(Messenger):
    """
    Implementation of sequential plate. Should not be used directly.
    """
    def __init__(self, name, size, indices, scale):
        self.name = name
        self.size = size
        self.indices = indices
        self._scale = scale
        self._counter = 0
        super().__init__()

    def __iter__(self):
        with ignore_jit_warnings([("Iterating over a tensor", RuntimeWarning)]), self:
            self._counter = 0
            for i in self.indices:
                self._counter += 1
                yield i if isinstance(i, Number) else i.item()

    def _pyro_sample(self, msg):
        frame = CondIndepStackFrame(self.name, None, self.size, self._counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        msg["scale"] = msg["scale"] * self._scale

    def _pyro_param(self, msg):
        frame = CondIndepStackFrame(self.name, None, self.size, self._counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        msg["scale"] = msg["scale"] * self._scale
