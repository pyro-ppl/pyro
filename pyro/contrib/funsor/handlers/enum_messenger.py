# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This file contains reimplementations of some of Pyro's core enumeration machinery,
which should eventually be drop-in replacements for the current versions.
"""
import functools
import math
from collections import OrderedDict

import torch
import funsor

import pyro.poutine.runtime
import pyro.poutine.util
from pyro.poutine.escape_messenger import EscapeMessenger
from pyro.poutine.subsample_messenger import _Subsample

from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.contrib.funsor.handlers.named_messenger import NamedMessenger
from pyro.contrib.funsor.handlers.replay_messenger import ReplayMessenger
from pyro.contrib.funsor.handlers.trace_messenger import TraceMessenger

funsor.set_backend("torch")


@functools.singledispatch
def _get_support_value(funsor_dist, name, **kwargs):
    raise ValueError("Could not extract point from {} at name {}".format(funsor_dist, name))


@_get_support_value.register(funsor.cnf.Contraction)
def _get_support_value_contraction(funsor_dist, name, **kwargs):
    delta_terms = [v for v in funsor_dist.terms
                   if isinstance(v, funsor.delta.Delta) and name in v.fresh]
    assert len(delta_terms) == 1
    return _get_support_value(delta_terms[0], name, **kwargs)


@_get_support_value.register(funsor.delta.Delta)
def _get_support_value_delta(funsor_dist, name, **kwargs):
    assert name in funsor_dist.fresh
    return OrderedDict(funsor_dist.terms)[name][0]


@_get_support_value.register(funsor.Tensor)
def _get_support_value_tensor(funsor_dist, name, **kwargs):
    assert name in funsor_dist.inputs
    return funsor.Tensor(
        funsor.ops.new_arange(funsor_dist.data, funsor_dist.inputs[name].size),
        OrderedDict([(name, funsor_dist.inputs[name])]),
        funsor_dist.inputs[name].size
    )


@_get_support_value.register(funsor.distribution.Distribution)
def _get_support_value_distribution(funsor_dist, name, expand=False):
    assert name == funsor_dist.value.name
    return funsor_dist.enumerate_support(expand=expand)


def _enum_strategy_default(dist, msg):
    sample_inputs = OrderedDict((f.name, funsor.Bint[f.size]) for f in msg["cond_indep_stack"]
                                if f.vectorized and f.name not in dist.inputs)
    sampled_dist = dist.sample(msg["name"], sample_inputs)
    return sampled_dist


def _enum_strategy_diagonal(dist, msg):
    sample_dim_name = "{}__PARTICLES".format(msg["name"])
    sample_inputs = OrderedDict({sample_dim_name: funsor.Bint[msg["infer"]["num_samples"]]})
    plate_names = frozenset(f.name for f in msg["cond_indep_stack"] if f.vectorized)
    ancestor_names = frozenset(k for k, v in dist.inputs.items() if v.dtype != 'real'
                               and k != msg["name"] and k not in plate_names)
    # TODO should the ancestor_indices be pyro.observed?
    ancestor_indices = {name: sample_dim_name for name in ancestor_names}
    sampled_dist = dist(**ancestor_indices).sample(
        msg["name"], sample_inputs if not ancestor_indices else None)
    if ancestor_indices:  # XXX is there a better way to account for this in funsor?
        sampled_dist = sampled_dist - math.log(msg["infer"]["num_samples"])
    return sampled_dist


def _enum_strategy_mixture(dist, msg):
    sample_dim_name = "{}__PARTICLES".format(msg["name"])
    sample_inputs = OrderedDict({sample_dim_name: funsor.Bint[msg['infer']['num_samples']]})
    plate_names = frozenset(f.name for f in msg["cond_indep_stack"] if f.vectorized)
    ancestor_names = frozenset(k for k, v in dist.inputs.items() if v.dtype != 'real'
                               and k != msg["name"] and k not in plate_names)
    plate_inputs = OrderedDict((k, dist.inputs[k]) for k in plate_names)
    # TODO should the ancestor_indices be pyro.sampled?
    ancestor_indices = {
        # TODO make this comprehension less gross
        name: _get_support_value(funsor.torch.distributions.CategoricalLogits(
            # sample different ancestors for each plate slice
            logits=funsor.Tensor(
                # TODO avoid use of torch.zeros here in favor of funsor.ops.new_zeros
                torch.zeros((1,)).expand(tuple(v.dtype for v in plate_inputs.values()) + (dist.inputs[name].dtype,)),
                plate_inputs
            ),
        )(value=name).sample(name, sample_inputs), name)
        for name in ancestor_names
    }
    sampled_dist = dist(**ancestor_indices).sample(
        msg["name"], sample_inputs if not ancestor_indices else None)
    if ancestor_indices:  # XXX is there a better way to account for this in funsor?
        sampled_dist = sampled_dist - math.log(msg["infer"]["num_samples"])
    return sampled_dist


def _enum_strategy_full(dist, msg):
    sample_dim_name = "{}__PARTICLES".format(msg["name"])
    sample_inputs = OrderedDict({sample_dim_name: funsor.Bint[msg["infer"]["num_samples"]]})
    sampled_dist = dist.sample(msg["name"], sample_inputs)
    return sampled_dist


def _enum_strategy_exact(dist, msg):
    if isinstance(dist, funsor.Tensor):
        dist = dist - dist.reduce(funsor.ops.logaddexp, msg["name"])
    return dist


def enumerate_site(dist, msg):
    # TODO come up with a better dispatch system for enumeration strategies
    if msg["infer"]["enumerate"] == "flat":
        return _enum_strategy_default(dist, msg)
    elif msg["infer"].get("num_samples", None) is None:
        return _enum_strategy_exact(dist, msg)
    elif msg["infer"]["num_samples"] > 1 and \
            (msg["infer"].get("expand", False) or msg["infer"].get("tmc") == "full"):
        return _enum_strategy_full(dist, msg)
    elif msg["infer"]["num_samples"] > 1 and msg["infer"].get("tmc", "diagonal") == "diagonal":
        return _enum_strategy_diagonal(dist, msg)
    elif msg["infer"]["num_samples"] > 1 and msg["infer"]["tmc"] == "mixture":
        return _enum_strategy_mixture(dist, msg)
    raise ValueError("{} not valid enum strategy".format(msg))


class EnumMessenger(NamedMessenger):
    """
    This version of EnumMessenger uses to_data to allocate a fresh enumeration dim
    for each discrete sample site.
    """
    def _pyro_sample(self, msg):
        if msg["done"] or msg["is_observed"] or \
                msg["infer"].get("enumerate") not in {"flat", "parallel"} or \
                isinstance(msg["fn"], _Subsample):
            return

        if "funsor" not in msg:
            msg["funsor"] = {}

        unsampled_log_measure = to_funsor(msg["fn"], output=funsor.Real)(value=msg["name"])
        msg["funsor"]["log_measure"] = enumerate_site(unsampled_log_measure, msg)
        msg["funsor"]["value"] = _get_support_value(
            msg["funsor"]["log_measure"], msg["name"], expand=msg["infer"].get("expand", False))
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
