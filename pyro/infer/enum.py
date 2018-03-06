from __future__ import absolute_import, division, print_function

import functools

from six.moves.queue import LifoQueue

from pyro import poutine
from pyro.infer.util import TreeSum
from pyro.poutine.trace import Trace


def _iter_discrete_filter(msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            msg["infer"].get("enumerate"))  # sequential or parallel


def _iter_discrete_escape(trace, msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            (msg["infer"].get("enumerate") == "sequential") and  # only sequential
            (msg["name"] not in trace))


def _iter_discrete_extend(trace, site, enum_tree):
    values = site["fn"].enumerate_support()
    log_probs = site["fn"].log_prob(values).detach()
    for i, (value, log_prob) in enumerate(zip(values, log_probs)):
        extended_site = site.copy()
        extended_site["value"] = value
        extended_trace = trace.copy()
        extended_trace.add_node(site["name"], **extended_site)
        extended_enum_tree = enum_tree.copy()
        extended_enum_tree.add(site["cond_indep_stack"], (i,))
        yield extended_trace, extended_enum_tree


def _iter_discrete_queue(graph_type, fn, *args, **kwargs):
    queue = LifoQueue()
    partial_trace = Trace()
    enum_tree = TreeSum()
    queue.put((partial_trace, enum_tree))
    while not queue.empty():
        partial_trace, enum_tree = queue.get()
        traced_fn = poutine.trace(poutine.escape(poutine.replay(fn, partial_trace),
                                                 functools.partial(_iter_discrete_escape, partial_trace)),
                                  graph_type=graph_type)
        try:
            yield traced_fn.get_trace(*args, **kwargs), enum_tree
        except poutine.util.NonlocalExit as e:
            e.reset_stack()
            for item in _iter_discrete_extend(traced_fn.trace, e.site, enum_tree):
                queue.put(item)


def iter_discrete_traces(graph_type, fn, *args, **kwargs):
    """
    Iterate over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this iterates over all choices.

    This yields traces scaled by the probability of the discrete choices made
    in the `trace`.

    :param str graph_type: The type of the graph, e.g. "flat" or "dense".
    :param callable fn: A stochastic function.
    :returns: An iterator over (weights, trace) pairs, where weights is a
        :class:`~pyro.infer.util.TreeSum` object.
    """
    already_counted = set()  # to avoid double counting
    for trace, enum_tree in _iter_discrete_queue(graph_type, fn, *args, **kwargs):
        # Collect log_probs for each iarange stack.
        log_probs = TreeSum()
        if not already_counted:
            log_probs.add((), 0)  # ensures globals are counted exactly once
        for name, site in trace.nodes.items():
            if _iter_discrete_filter(site):
                cond_indep_stack = site["cond_indep_stack"]
                log_prob = site["fn"].log_prob(site["value"]).detach()
                log_probs.add(cond_indep_stack, log_prob)

        # Avoid double-counting across traces.
        weights = log_probs.exp()
        for context in enum_tree.items():
            if context in already_counted:
                cond_indep_stack, _ = context
                weights.prune(cond_indep_stack)
            else:
                already_counted.add(context)

        yield weights, trace


def _config_enumerate(default):

    def config_fn(site):
        if site["type"] != "sample" or site["is_observed"]:
            return {}
        if not getattr(site["fn"], "enumerable", False):
            return {}
        if "enumerate" in site["infer"]:
            return {}  # do not overwrite existing config
        return {"enumerate": default}

    return config_fn


def config_enumerate(guide=None, default="sequential"):
    """
    Configures each enumerable site a guide to enumerate with given method,
    ``site["infer"]["enumerate"] = default``. This can be used as either a
    function::

        guide = config_enumerate(guide)

    or as a decorator::

        @config_enumerate
        def guide1(*args, **kwargs):
            ...

        @config_enumerate(default="parallel")
        def guide2(*args, **kwargs):
            ...

    This does not overwrite existing annotations ``infer={"enumerate": ...}``.

    :param callable guide: a pyro model that will be used as a guide in
        :class:`~pyro.infer.svi.SVI`.
    :param str default: one of "sequential", "parallel", or None.
    :return: an annotated guide
    :rtype: callable
    """
    if default not in ["sequential", "parallel", None]:
        raise ValueError("Invalid default value. Expected 'sequential', 'parallel', or None, but got {}".format(
            repr(default)))
    # Support usage as a decorator:
    if guide is None:
        return lambda guide: config_enumerate(guide, default=default)

    return poutine.infer_config(guide, _config_enumerate(default))
