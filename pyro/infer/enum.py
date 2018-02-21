from __future__ import absolute_import, division, print_function

import math

import torch
from six.moves.queue import LifoQueue
from torch.autograd import Variable

from pyro import poutine
from pyro.distributions.util import sum_rightmost
from pyro.poutine.trace import Trace


def _iter_discrete_filter(name, msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            (msg["infer"].get("enumerate") == "sequential"))


def _iter_discrete_escape(trace, msg):
    return _iter_discrete_filter(msg["name"], msg) and (msg["name"] not in trace)


def iter_discrete_traces(graph_type, max_iarange_nesting, fn, *args, **kwargs):
    """
    Iterate over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this iterates over all choices.

    This yields `(scale, trace)` pairs, where `scale` is the probability of the
    discrete choices made in the `trace`.

    :param str graph_type: The type of the graph, e.g. "flat" or "dense".
    :param callable fn: A stochastic function.
    :returns: An iterator over (scale, trace) pairs.
    """
    queue = LifoQueue()
    queue.put(Trace())
    q_fn = poutine.queue(fn, queue=queue)
    while not queue.empty():
        q_fn = poutine.queue(fn, queue=queue, escape_fn=_iter_discrete_escape)
        full_trace = poutine.trace(q_fn, graph_type=graph_type).get_trace(*args, **kwargs)

        # Scale trace by probability of discrete choices.
        log_pdf = 0
        full_trace.compute_batch_log_pdf(site_filter=_iter_discrete_filter)
        for name, site in full_trace.nodes.items():
            if _iter_discrete_filter(name, site):
                log_pdf += sum_rightmost(site["batch_log_pdf"], max_iarange_nesting)
        if isinstance(log_pdf, Variable):
            scale = torch.exp(log_pdf.detach())
        else:
            scale = math.exp(log_pdf)

        yield scale, full_trace


def _enumerate_discrete(default):

    def config_fn(site):
        if site["type"] != "sample" or site["is_observed"]:
            return {}
        if not site["fn"].get("enumerable", False):
            return {}
        if "enumerate" in site["infer"]:
            return {}  # do not override existing config
        return {"enumerate": default}

    return config_fn


def enumerate_discrete(guide=None, default="sequential"):
    """
    Configures each enumerable site a guide to enumerate with given method,
    ``site["infer"]["enumerate"] = default``. This can be used as either a
    function::

        guide = enumerate_discrete(guide)

    or as a decorator::

        @enumerate_discrete
        def guide1(*args, **kwargs):
            ...

        @enumerate_discrete(default="parallel")
        def guide2(*args, **kwargs):
            ...

    :param callable guide: a pyro model that will be used as a guide in
        :class:`~pyro.infer.svi.SVI`.
    :param str default: either "sequential" or "parallel".
    :return: an annotated guide
    :rtype: callable
    """
    # Support usage as a decorator:
    if guide is None:
        return lambda guide: enumerate_discrete(guide, default=default)

    return poutine.infer_config(guide, _enumerate_discrete(default))
