import functools
from six.moves.queue import LifoQueue

import torch

from pyro import util
from pyro import poutine
from pyro.poutine.trace import Trace


def site_is_discrete(name, site):
    return getattr(site, "enumerable", False)


def iter_discrete_traces(poutine_trace, fn, *args, **kwargs):
    """
    Iterate over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this iterates over all choices.

    This yields `(scale, trace)` pairs, where `scale` is the probability of the
    discrete choices made in the `trace`.

    :param callable poutine_trace: A trace poutine, either `poutine.trace` or
        `poutine.tracegraph`.
    :param callable fn: A stochastic function.
    :returns: An iterator over (scale, trace) pairs.
    """
    queue = LifoQueue()
    queue.put(Trace())
    while not queue.empty():
        partial_trace = queue.get()
        escape_fn = functools.partial(util.discrete_escape, partial_trace)
        traced_fn = poutine_trace(poutine.escape(poutine.replay(fn, partial_trace), escape_fn))
        try:
            full_trace = traced_fn.get_trace(*args, **kwargs)
        except util.NonlocalExit as e:
            for extended_trace in util.enum_extend(traced_fn.trace.copy(), e.site):
                # TODO Scale traces by the choice probability.
                queue.put(extended_trace)
            continue

        # Scale trace by probability of discrete choices.
        log_pdf = full_trace.log_pdf(site_filter=site_is_discrete).detach()
        scale = torch.exp(log_pdf).data[0]
        yield scale, full_trace


def scale_trace(trace, scale):
    """
    Scale all sample and observe sites in a trace (copies the trace).

    :param Trace trace: A pyro trace.
    :param scale: A nonnegative scaling constant.
    :type scale: float or torch.Tensor or torch.autograd.Variable
    :returns: A scaled copy of the trace.
    :rtype: Trace
    """
    trace = trace.copy()
    for name, site in trace.items():
        if "scale" in site:
            site = site.copy()
            trace[name] = site
            site["scale"] = site["scale"] * scale
        # Clear memoized computations.
        if site["type"] in ("observe", "sample"):
            site.pop("log_pdf", None)
            site.pop("batch_log_pdf", None)
    return trace
