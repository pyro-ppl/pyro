# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This file contains reimplementations of some of Pyro's core enumeration machinery,
which should eventually be drop-in replacements for the current versions.
"""
import functools
from collections import OrderedDict

import funsor

import pyro.poutine.runtime
import pyro.poutine.util
from pyro.poutine.escape_messenger import EscapeMessenger
from pyro.poutine.subsample_messenger import _Subsample

from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.contrib.funsor.handlers.named_messenger import BaseEnumMessenger
from pyro.contrib.funsor.handlers.replay_messenger import ReplayMessenger
from pyro.contrib.funsor.handlers.trace_messenger import TraceMessenger

funsor.set_backend("torch")


def _enum_strategy_enum(msg):
    dist = to_funsor(msg["fn"], output=funsor.reals())(value=msg['name'])
    raw_value = msg["fn"].enumerate_support(expand=msg["infer"].get("expand", False))
    size = raw_value.numel()
    funsor_value = funsor.Tensor(
        raw_value.squeeze(),
        OrderedDict([(msg["name"], funsor.bint(size))]),
        dist.inputs[msg["name"]].dtype
    )
    if isinstance(dist, funsor.Tensor):
        # ensure dist is normalized
        dist = dist - dist.reduce(funsor.ops.logaddexp, msg['name'])
    return dist, funsor_value


def enumerate_site(msg):
    # TODO come up with a better dispatch system for enumeration strategies
    if msg["infer"].get("num_samples", None) is None:
        return _enum_strategy_enum(msg)
    # TODO restore Monte Carlo strategies
    raise ValueError("{} not valid enum strategy".format(msg))


class EnumMessenger(BaseEnumMessenger):
    """
    This version of EnumMessenger uses to_data to allocate a fresh enumeration dim
    for each discrete sample site.
    """
    def _pyro_sample(self, msg):
        if msg["done"] or msg["is_observed"] or msg["infer"].get("enumerate") != "parallel" \
                or isinstance(msg["fn"], _Subsample):
            return

        if "funsor" not in msg:
            msg["funsor"] = {}
        msg["funsor"]["log_measure"], msg["funsor"]["value"] = enumerate_site(msg)
        msg["value"] = to_data(msg["funsor"]["value"])
        msg["done"] = True


def queue(fn=None, queue=None,
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
