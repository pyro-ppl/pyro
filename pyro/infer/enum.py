from __future__ import absolute_import, division, print_function

from six.moves.queue import LifoQueue

from pyro import poutine
from pyro.poutine import Trace


def _iter_discrete_escape(trace, msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            (msg["infer"].get("enumerate") == "sequential") and  # only sequential
            (msg["name"] not in trace))


def _iter_discrete_extend(trace, site, **ignored):
    values = site["fn"].enumerate_support()
    for i, value in enumerate(values):
        extended_site = site.copy()
        extended_site["infer"] = site["infer"].copy()
        extended_site["infer"]["_enum_total"] = len(values)
        extended_site["value"] = value
        extended_trace = trace.copy()
        extended_trace.add_node(site["name"], **extended_site)
        yield extended_trace


def iter_discrete_traces(graph_type, fn, *args, **kwargs):
    """
    Iterate over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this iterates over all choices.

    This yields traces scaled by the probability of the discrete choices made
    in the `trace`.

    :param str graph_type: The type of the graph, e.g. "flat" or "dense".
    :param callable fn: A stochastic function.
    :returns: An iterator over traces pairs.
    """
    queue = LifoQueue()
    queue.put(Trace())
    traced_fn = poutine.trace(
        poutine.queue(fn, queue, escape_fn=_iter_discrete_escape, extend_fn=_iter_discrete_extend),
        graph_type=graph_type)
    while not queue.empty():
        yield traced_fn.get_trace(*args, **kwargs)


def _config_enumerate(default):

    def config_fn(site):
        if site["type"] != "sample" or site["is_observed"]:
            return {}
        if not getattr(site["fn"], "has_enumerate_support", False):
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

    return poutine.infer_config(guide, config_fn=_config_enumerate(default))
