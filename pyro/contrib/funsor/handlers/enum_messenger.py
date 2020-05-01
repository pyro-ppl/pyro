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
from pyro.contrib.funsor.handlers.named_messenger import BaseEnumMessenger, LocalNamedMessenger
from pyro.contrib.funsor.handlers.replay_messenger import ReplayMessenger
from pyro.contrib.funsor.handlers.trace_messenger import TraceMessenger

funsor.set_backend("torch")


class MarkovMessenger(LocalNamedMessenger):
    """
    LocalNamedMessenger is meant to be a drop-in replacement for pyro.markov.
    """
    pass


def _get_delta_point(funsor_dist, name):
    assert isinstance(funsor_dist, funsor.terms.Funsor)
    assert name in funsor_dist.inputs
    if isinstance(funsor_dist, funsor.delta.Delta):
        return OrderedDict(funsor_dist.terms)[name][0]
    elif isinstance(funsor_dist, funsor.cnf.Contraction):
        delta_terms = [v for v in funsor_dist.terms
                       if isinstance(v, funsor.delta.Delta) and name in v.fresh]
        assert len(delta_terms) == 1
        return _get_delta_point(delta_terms[0], name)
    elif isinstance(funsor_dist, funsor.Tensor):
        return funsor_dist
    else:
        raise ValueError("Could not extract point from {} at name {}".format(funsor_dist, name))


def _enum_strategy_diagonal(msg):
    dist = to_funsor(msg["fn"], output=funsor.reals())(value=msg['name'])
    sample_dim_name = "{}__PARTICLES".format(msg['name'])
    sample_inputs = OrderedDict({sample_dim_name: funsor.bint(msg["infer"]["num_samples"])})
    plate_names = frozenset(f.name for f in msg["cond_indep_stack"] if f.vectorized)
    ancestor_names = frozenset(k for k, v in dist.inputs.items() if v.dtype != 'real'
                               and k != msg['name'] and k not in plate_names)
    # TODO should the ancestor_indices be pyro.observed?
    ancestor_indices = {name: sample_dim_name for name in ancestor_names}
    sampled_dist = dist(**ancestor_indices).sample(
        msg['name'], sample_inputs if not ancestor_indices else None)
    return sampled_dist, _get_delta_point(sampled_dist, msg['name'])


def _enum_strategy_mixture(msg):
    dist = to_funsor(msg["fn"], output=funsor.reals())(value=msg['name'])
    sample_dim_name = "{}__PARTICLES".format(msg['name'])
    sample_inputs = OrderedDict({sample_dim_name: funsor.bint(msg['infer']['num_samples'])})
    plate_names = frozenset(f.name for f in msg["cond_indep_stack"] if f.vectorized)
    ancestor_names = frozenset(k for k, v in dist.inputs.items() if v.dtype != 'real'
                               and k != msg['name'] and k not in plate_names)
    plate_inputs = OrderedDict((k, dist.inputs[k]) for k in plate_names)
    # TODO should the ancestor_indices be pyro.sampled?
    ancestor_indices = {
        name: funsor.distributions.Categorical(
            # sample different ancestors for each plate slice
            logits=funsor.Tensor(
                funsor.ops.new_zeros(1).expand(tuple(plate_inputs.values()) + (dist.inputs[name].dtype,)),
                plate_inputs
            ),
        )(value=name).sample(name, sample_inputs)
        for name in ancestor_names
    }
    sampled_dist = dist(**ancestor_indices).sample(
        msg['name'], sample_inputs if not ancestor_indices else None)
    return sampled_dist, _get_delta_point(sampled_dist, msg['name'])


def _enum_strategy_full(msg):
    dist = to_funsor(msg["fn"], output=funsor.reals())(value=msg['name'])
    sample_dim_name = "{}__PARTICLES".format(msg['name'])
    sample_inputs = OrderedDict({sample_dim_name: funsor.bint(msg["infer"]["num_samples"])})
    sampled_dist = dist.sample(msg['name'], sample_inputs)
    return sampled_dist, _get_delta_point(sampled_dist, msg['name'])


def _enum_strategy_enum(msg):
    dist = to_funsor(msg["fn"], output=funsor.reals())(value=msg['name'])
    raw_value = msg["fn"].enumerate_support(expand=msg["infer"].get("expand", False))
    size = raw_value.numel()
    funsor_value = funsor.Tensor(
        raw_value.squeeze(),
        OrderedDict([(msg["name"], funsor.bint(size))]),
        size
    )
    return dist, funsor_value


def enumerate_site(msg):
    # TODO come up with a better dispatch system for enumeration strategies
    if msg["infer"].get("num_samples", None) is None:
        return _enum_strategy_enum(msg)
    elif msg["infer"]["num_samples"] > 1 and \
            (msg["infer"].get("expand", False) or msg["infer"].get("tmc") == "full"):
        return _enum_strategy_full(msg)
    elif msg["infer"]["num_samples"] > 1 and msg["infer"].get("tmc", "diagonal") == "diagonal":
        return _enum_strategy_diagonal(msg)
    elif msg["infer"]["num_samples"] > 1 and msg["infer"]["tmc"] == "mixture":
        return _enum_strategy_mixture(msg)
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

        msg["infer"]["funsor_log_measure"], msg["infer"]["funsor_value"] = enumerate_site(msg)
        msg["value"] = to_data(msg["infer"]["funsor_value"])
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
