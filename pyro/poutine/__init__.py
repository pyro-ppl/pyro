# poutines
from .block_poutine import BlockPoutine
from .poutine import Poutine  # noqa: F401
from .queue_poutine import QueuePoutine
from .replay_poutine import ReplayPoutine
from .trace_poutine import TracePoutine
from .tracegraph_poutine import TraceGraphPoutine
from .lift_poutine import LiftPoutine

# trace data structures
from .trace import Trace, TraceGraph  # noqa: F401


############################################
# Begin primitive operations
############################################


def trace(fn):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :returns: stochastic function wrapped in a TracePoutine
    :rtype: pyro.poutine.TracePoutine

    Alias for TracePoutine constructor.

    Given a callable that contains Pyro primitive calls, return a TracePoutine callable
    that records the inputs and outputs to those primitive calls.
    Adds trace data structure site constructors to primitive stacks
    """
    return TracePoutine(fn)


def tracegraph(fn):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :returns: stochastic function wrapped in a TraceGraphPoutine
    :rtype: pyro.poutine.TraceGraphPoutine

    Alias for TraceGraphPoutine constructor.

    Given a callable that contains Pyro primitive calls,, return a TraceGraphPoutine callable
    that records the inputs and outputs to those primitive calls and their dependencies.
    Adds trace and tracegraph data structure site constructors to primitive stacks
    """
    return TraceGraphPoutine(fn)


def replay(fn, trace, sites=None):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param trace: a Trace data structure to replay against
    :param sites: list or dict of names of sample sites in fn to replay against,
    defaulting to all sites
    :returns: stochastic function wrapped in a ReplayPoutine
    :rtype: pyro.poutine.ReplayPoutine

    Alias for ReplayPoutine constructor.

    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace
    """
    return ReplayPoutine(fn, trace, sites=sites)


def block(fn, hide=None, expose=None, hide_types=None, expose_types=None):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param hide: list of site names to hide
    :param expose: list of site names to be exposed while all others hidden
    :param hide_types: list of site types to be hidden
    :param expose_types: list of site types to be exposed while all others hidden
    :returns: stochastic function wrapped in a BlockPoutine
    :rtype: pyro.poutine.BlockPoutine

    Alias for BlockPoutine constructor.

    Given a callable that contains Pyro primitive calls,
    selectively hide some of those calls from poutines higher up the stack
    """
    return BlockPoutine(fn, hide=hide, expose=expose,
                        hide_types=hide_types, expose_types=expose_types)

def lift(fn, prior):
    """
    Given a stochastic function with param calls and a prior distribution,
    create a stochastic function where all param calls are replaced by sampling from prior
    prior should be a callable with the same signature as pyro.param
    """
    def _fn(*args, **kwargs):
        p = LiftPoutine(fn, prior)
        return p(*args, **kwargs)
    return _fn


def queue(fn, queue=None, max_tries=None):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param queue: a queue data structure like multiprocessing.Queue to hold partial traces
    :param max_tries: maximum number of attempts to compute a single complete trace
    :returns: stochastic function wrapped in a QueuePoutine
    :rtype: pyro.poutine.QueuePoutine
    Alias for QueuePoutine constructor.
    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue
    """
    return QueuePoutine(fn, queue=queue, max_tries=max_tries)
