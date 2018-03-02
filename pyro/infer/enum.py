from __future__ import absolute_import, division, print_function

import functools

from six.moves.queue import LifoQueue

from pyro import poutine
from pyro.infer.util import MultiViewTensor
from pyro.poutine.trace import Trace


def _iter_discrete_escape(trace, msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            (msg["infer"].get("enumerate") == "sequential") and  # only sequential
            (msg["name"] not in trace))


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
            full_trace = traced_fn.get_trace(*args, **kwargs)
        except poutine.util.NonlocalExit as e:
            e.reset_stack()
            for i, value in enumerate(e.site["fn"].enumerate_support()):
                site = e.site.copy()
                site["value"] = value
                extended_trace = partial_trace.copy()
                extended_trace.add_node(site["name"], **site)
                queue.put((enum_stack + (i,), extended_trace))
            continue

        # Compute total log probability of trace.
        log_prob = MultiViewTensor()  # is this right?
        for name, site in full_trace.nodes.items():
            # find sample sites that are enumerated either sequentially or in parallel
            if site["type"] == "sample" and not site["is_observed"] and site["infer"].get("enumerate"):
                log_prob.add(site["fn"].log_prob(site["value"]).detach())

        # Scale sites by probability of discrete choices.
        if log_prob:
            for name, site in full_trace.nodes.items():
                if site["type"] == "sample":
                    value_shape = site["value"].shape
                    event_shape = getattr(site["fn"], "event_shape", value_shape)
                    log_prob_shape = value_shape[:len(value_shape) - len(event_shape)]
                    site["scale"] = site["scale"] * log_prob.contract(log_prob_shape).exp()
                    assert "log_pdf" not in site, "site contains stale log_pdf"

        yield full_trace


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
