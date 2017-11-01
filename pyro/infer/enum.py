import torch
from torch.autograd import Variable

import pyro
from pyro import poutine, util
from pyro.poutine import LambdaPoutine
from pyro.poutine.trace import Trace
from six.moves.queue import LifoQueue


def site_is_discrete(name, site):
    return getattr(site["fn"], "enumerable", False)


# Batched .enumerate_support() assumes batched values are independent.
# This wraps discrete_escape to also check that every batched escape site occurs within an iarange.
def make_escape_fn(trace):

    def escape_fn(msg):
        if not util.discrete_escape(trace, msg):
            return False
        batch_shape = msg["fn"].batch_shape(msg["value"], *msg["args"], **msg["kwargs"])
        is_batched = any(size > 1 for size in batch_shape)
        inside_iarange = any(frame.vectorized for frame in msg["map_data_stack"])
        if is_batched and not inside_iarange:
            raise ValueError(
                    "poutine.queue encountered a batched pyro.sample site '{}' outiside of a pyro.iarange. "
                    "To fix, either enclose in a pyro.iarange, or avoid batching.".format(msg["name"]))
        return True

    return escape_fn


def iter_discrete_traces(graph_type, fn, *args, **kwargs):
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
    while not queue.empty():
        partial_trace = queue.get()
        escape_fn = make_escape_fn(partial_trace)
        traced_fn = poutine.trace(poutine.escape(poutine.replay(fn, partial_trace), escape_fn),
                                  graph_type=graph_type)
        try:
            full_trace = traced_fn.get_trace(*args, **kwargs)
        except util.NonlocalExit as e:
            for extended_trace in util.enum_extend(traced_fn.trace.copy(), e.site):
                queue.put(extended_trace)
            continue

        # Scale trace by probability of discrete choices.
        log_pdf = full_trace.batch_log_pdf(site_filter=site_is_discrete)
        if isinstance(log_pdf, float):
            log_pdf = torch.Tensor([log_pdf])
        if isinstance(log_pdf, torch.Tensor):
            log_pdf = Variable(log_pdf)
        scale = torch.exp(log_pdf.detach())
        yield scale, full_trace
