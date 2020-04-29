# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""
This file contains reimplementations of some of Pyro's core enumeration machinery,
which should eventually be drop-in replacements for the current versions.
"""
import functools
from queue import LifoQueue
from collections import OrderedDict

import funsor

import pyro.poutine.runtime
import pyro.poutine.util

from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.escape_messenger import EscapeMessenger
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.replay_messenger import ReplayMessenger as OrigReplayMessenger
from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.subsample_messenger import SubsampleMessenger as OrigSubsampleMessenger
from pyro.poutine.trace_messenger import TraceMessenger as OrigTraceMessenger

from pyro.contrib.funsor import to_funsor, to_data
from pyro.contrib.funsor.named_messenger import DimType, \
    BaseEnumMessenger, GlobalNamedMessenger, LocalNamedMessenger, NamedMessenger

funsor.set_backend("torch")


class MarkovMessenger(LocalNamedMessenger):
    """
    LocalNamedMessenger is meant to be a drop-in replacement for pyro.markov.
    """
    pass


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
        if indices is None:
            indices = funsor.ops.new_arange(funsor.tensor.get_default_prototype(), self.size)
        assert len(indices) == size

        self._indices = funsor.Tensor(
            indices, OrderedDict([(self.name, funsor.bint(self.size))]), self.size
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


class SubsampleMessenger(IndepMessenger):

    def __init__(self, name, size=None, subsample_size=None, subsample=None, dim=None,
                 use_cuda=None, device=None):
        size, subsample_size, indices = OrigSubsampleMessenger._subsample(
            name, size, subsample_size, subsample, use_cuda, device)
        super().__init__(name, subsample_size, dim, indices)
        self.subsample_size = subsample_size
        self._full_size = size
        self._scale = size / subsample_size

    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        msg["scale"] = msg["scale"] * self._scale

    def _pyro_param(self, msg):
        super()._pyro_param(msg)
        msg["scale"] = msg["scale"] * self._scale


class SequentialPlateMessenger(LocalNamedMessenger):
    def __init__(self, name=None, size=None, dim=None):
        self.name, self.size, self.dim, self.counter = name, size, dim, 0
        super().__init__(history=0, keep=False)

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


class EnumMessenger(BaseEnumMessenger):
    """
    This version of EnumMessenger uses to_data to allocate a fresh enumeration dim
    for each discrete sample site.
    """
    def _pyro_sample(self, msg):

        if msg["done"] or msg["is_observed"] or msg["infer"].get("enumerate") != "parallel":
            return

        if msg["infer"].get("num_samples", None) is not None:
            raise NotImplementedError("TODO implement multiple sampling")

        if msg["infer"].get("expand", False):
            raise NotImplementedError("expand=True not implemented")

        msg["infer"]["funsor_log_measure"] = to_funsor(msg["fn"], funsor.reals())(value=msg["name"])
        raw_value = msg["fn"].enumerate_support(expand=False).squeeze()
        size = raw_value.numel()
        msg["infer"]["funsor_value"] = funsor.Tensor(
            raw_value,
            OrderedDict([(msg["name"], funsor.bint(size))]),
            size
        )

        msg["value"] = to_data(msg["infer"]["funsor_value"])
        msg["done"] = True


class TraceMessenger(OrigTraceMessenger):
    """
    This version of TraceMessenger does its packing online instead of after the fact,
    converting all distributions and values to Funsors as soon as they are available.
    """
    def _pyro_post_sample(self, msg):
        if isinstance(msg["fn"], _Subsample):
            return super()._pyro_post_sample(msg)
        if "funsor_fn" not in msg["infer"]:
            msg["infer"]["funsor_fn"] = to_funsor(msg["fn"], funsor.reals())
        if "funsor_log_prob" not in msg["infer"]:
            msg["infer"]["funsor_log_prob"] = to_funsor(msg["fn"].log_prob(msg["value"]),
                                                        funsor.reals())
        if "funsor_value" not in msg["infer"]:
            value_output = funsor.reals(*getattr(msg["fn"], "event_shape", ()))
            msg["infer"]["funsor_value"] = to_funsor(msg["value"], value_output)
        return super()._pyro_post_sample(msg)


class PackTraceMessenger(OrigTraceMessenger):
    """
    This version of TraceMessenger records information necessary to do packing after execution.
    Each sample site is annotated with a "dim_to_name" dictionary,
    which can be passed directly to funsor.to_funsor.
    """
    def _pyro_post_sample(self, msg):
        if isinstance(msg["fn"], _Subsample):
            return super()._pyro_post_sample(msg)
        msg["infer"]["dim_to_name"] = NamedMessenger._get_dim_to_name(msg["fn"].batch_shape)
        msg["infer"]["dim_to_name"].update(NamedMessenger._get_dim_to_name(
            msg["value"].shape[:len(msg["value"]).shape - len(msg["fn"].event_shape)]))
        return super()._pyro_post_sample(msg)


class ReplayMessenger(OrigReplayMessenger):
    """
    This version of ReplayMessenger is almost identical to the original version,
    except that it calls to_data on the replayed funsor values.
    This may result in different unpacked shapes, but should produce correct allocations.
    """
    def _pyro_sample(self, msg):
        name = msg["name"]
        if self.trace is not None and name in self.trace:
            guide_msg = self.trace.nodes[name]
            if msg["is_observed"]:
                return None
            if guide_msg["type"] != "sample" or guide_msg["is_observed"]:
                raise RuntimeError("site {} must be sample in trace".format(name))
            # TODO make this work with sequential enumeration
            if guide_msg["infer"].get("funsor_value", None) is not None:
                msg["value"] = to_data(guide_msg["infer"]["funsor_value"])  # only difference is here
            else:
                msg["value"] = guide_msg["value"]
            msg["infer"] = guide_msg["infer"]
            msg["done"] = True


def enum_seq(fn=None, queue=None,
             max_tries=int(1e6), num_samples=-1,
             extend_fn=pyro.poutine.util.enum_extend,
             escape_fn=pyro.poutine.util.discrete_escape):
    """
    Used in sequential enumeration over discrete variables (copied from poutine.queue).

    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param q: a queue data structure like multiprocessing.Queue to hold partial traces
    :param max_tries: maximum number of attempts to compute a single complete trace
    :param extend_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a list of extended traces
    :param escape_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a boolean value to decide whether to exit
    :param num_samples: optional number of extended traces for extend_fn to return
    :returns: stochastic function decorated with poutine logic
    """
    # TODO rewrite this to use purpose-built trace/replay handlers
    if queue is None:
        queue = LifoQueue()

    def wrapper(wrapped):
        def _fn(*args, **kwargs):

            for i in range(max_tries):
                assert not queue.empty(), \
                    "trying to get() from an empty queue will deadlock"

                next_trace = queue.get()
                try:
                    ftr = TraceMessenger()(
                        EscapeMessenger(escape_fn=functools.partial(escape_fn, next_trace))(
                            ReplayMessenger(trace=next_trace)(wrapped)))
                    return ftr(*args, **kwargs)
                except pyro.poutine.runtime.NonlocalExit as site_container:
                    site_container.reset_stack()  # TODO implement missing ._reset()s
                    for tr in extend_fn(ftr.trace.copy(), site_container.site,
                                        num_samples=num_samples):
                        queue.put(tr)

            raise ValueError("max tries ({}) exceeded".format(str(max_tries)))
        return _fn

    return wrapper(fn) if fn is not None else wrapper
