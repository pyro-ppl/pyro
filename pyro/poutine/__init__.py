from .block_poutine import BlockPoutine
from .poutine import Poutine  # noqa: F401
from .queue_poutine import QueuePoutine
from .replay_poutine import ReplayPoutine
from .trace import Trace  # noqa: F401
from .trace_poutine import TracePoutine
from .tracegraph_poutine import TraceGraphPoutine


############################################
# Begin primitive operations
# XXX should these be returned as Poutines?
############################################

def trace(fn):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that records the inputs and outputs to those primitive calls
    Adds trace data structure site constructors to primitive stacks

    tr = trace(fn)(*args, **kwargs)
    """
    return TracePoutine(fn)


def tracegraph(fn, graph_output=None):

    return TraceGraphPoutine(fn)


def replay(fn, trace, sites=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace

    ret = replay(fn, trace, sites=some_sites)(*args, **kwargs)
    """
    return ReplayPoutine(fn, trace, sites=sites)


def block(fn, hide=None, expose=None, hide_types=None, expose_types=None):
    """
    Given a callable that contains Pyro primitive calls,
    hide the primitive calls at sites

    ret = block(fn, hide=["a"], expose=["b"])(*args, **kwargs)

    Also expose()?
    """
    return BlockPoutine(fn, hide=hide, expose=expose,
                        hide_types=hide_types, expose_types=expose_types)


def queue(fn, queue=None, max_tries=None):
    """
    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue
    """

    return QueuePoutine(fn, queue=queue, max_tries=max_tries)
