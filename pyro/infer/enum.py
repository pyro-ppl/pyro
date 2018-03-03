from __future__ import absolute_import, division, print_function

import functools

from six.moves.queue import LifoQueue

from pyro import poutine
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


def iter_discrete_traces(graph_type, max_iarange_nesting, fn, *args, **kwargs):
    """
    Iterate over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this iterates over all choices.

    This yields traces scaled by the probability of the discrete choices made
    in the `trace`.

    :param str graph_type: The type of the graph, e.g. "flat" or "dense".
    :param callable fn: A stochastic function.
    :returns: An iterator over scaled traces.
    """
    already_counted = set()
    for trace in _iter_discrete_queue(graph_type, fn, *args, **kwargs):
        log_prob = 0
        scale = None
        enum_stack = ()
        for name, site in trace.nodes.items():
            if site["type"] != "sample":
                continue

            if "enum_log_prob" in site["infer"]:
                log_prob = log_prob + site["infer"]["enum_log_prob"]
                scale = log_prob.exp()

            if scale is None:
                continue

            # Avoid double counting in sequential enumeration.
            enum_stack = site["infer"].get("enum_stack", enum_stack)
            if (name, enum_stack) in already_counted:
                site["infer"]["enum_scale"] = 0
            else:
                already_counted.add((name, enum_stack))
                site["infer"]["enum_scale"] = scale

        yield trace


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
