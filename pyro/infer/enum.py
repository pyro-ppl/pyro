from __future__ import absolute_import, division, print_function

import functools

from six.moves.queue import LifoQueue

from pyro import poutine
from pyro.infer.util import TensorTree
from pyro.poutine.trace import Trace


def _iter_discrete_escape(trace, msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            (msg["infer"].get("enumerate") == "sequential") and  # only sequential
            (msg["name"] not in trace))


def _iter_discrete_extend(trace, site, enum_stack):
    values = site["fn"].enumerate_support()
    log_probs = site["fn"].log_prob(values).detach()
    for i, (value, log_prob) in enumerate(zip(values, log_probs)):
        extended_enum_stack = enum_stack + (i,)
        extended_site = site.copy()
        extended_site["value"] = value
        extended_site["infer"] = site["infer"].copy()
        extended_site["infer"]["enum_stack"] = extended_enum_stack
        extended_site["infer"]["enum_log_prob"] = log_prob
        extended_trace = trace.copy()
        extended_trace.add_node(site["name"], **extended_site)
        yield extended_enum_stack, extended_trace


def _iter_discrete_queue(graph_type, fn, *args, **kwargs):
    queue = LifoQueue()
    enum_stack = ()
    partial_trace = Trace()
    queue.put((enum_stack, partial_trace))
    while not queue.empty():
        enum_stack, partial_trace = queue.get()
        traced_fn = poutine.trace(poutine.escape(poutine.replay(fn, partial_trace),
                                                 functools.partial(_iter_discrete_escape, partial_trace)),
                                  graph_type=graph_type)
        try:
            yield traced_fn.get_trace(*args, **kwargs)
        except poutine.util.NonlocalExit as e:
            e.reset_stack()
            for item in _iter_discrete_extend(traced_fn.trace, e.site, enum_stack):
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
    :returns: An iterator over (log_prob, trace) pairs.
    """
    already_counted = set()  # to avoid double counting
    for trace in _iter_discrete_queue(graph_type, fn, *args, **kwargs):
        # Collect log_probs for each iarange stack.
        log_probs = TensorTree()
        to_prune = set()
        for name, site in trace.nodes.items():
            if site["type"] == "sample" and "enum_stack" in site["infer"]:
                cond_indep_stack = tuple(site["cond_indep_stack"])
                log_probs.add(cond_indep_stack, site["infer"]["enum_log_prob"])

                # Avoid double counting.
                context = cond_indep_stack, site["infer"]["enum_stack"]
                if context in already_counted:
                    to_prune.add(cond_indep_stack)
                else:
                    already_counted.add(context)
        for cond_indep_stack in to_prune:
            log_probs.prune_upstream(cond_indep_stack)

        weights = log_probs.exp()
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
