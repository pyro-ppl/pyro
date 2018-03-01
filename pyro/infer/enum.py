from __future__ import absolute_import, division, print_function

from six.moves.queue import LifoQueue

from pyro import poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.util import reduce_to_shape
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

    This yields `(scale, trace)` pairs, where `scale` is the probability of the
    discrete choices made in the `trace`.

    :param str graph_type: The type of the graph, e.g. "flat" or "dense".
    :param callable fn: A stochastic function.
    :returns: An iterator over scaled traces.
    """
    queue = LifoQueue()
    queue.put(Trace())
    q_fn = poutine.queue(fn, queue=queue, escape_fn=_iter_discrete_escape)
    while not queue.empty():
        full_trace = poutine.trace(q_fn, graph_type=graph_type).get_trace(*args, **kwargs)

        # Compute total log probability of trace.
        log_prob = 0
        for name, site in full_trace.nodes.items():
            # find sample sites that are enumerated either sequentially or in parallel
            if site["type"] == "sample" and not site["is_observed"] and site["infer"].get("enumerate"):
                log_prob = log_prob + site["fn"].log_prob(site["value"]).detach()

        # Scale sites by probability of discrete choices.
        if not is_identically_zero(log_prob):
            prob = log_prob.exp()
            for name, site in full_trace.nodes.items():
                if site["type"] == "sample":
                    shape = site["value"].shape
                    event_shape = getattr(site["fn"], "event_shape", shape)
                    if event_shape:
                        shape = shape[:-len(event_shape)]
                    site["scale"] = site["scale"] * reduce_to_shape(prob, shape)

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
