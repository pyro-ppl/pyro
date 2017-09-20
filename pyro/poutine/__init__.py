from pyro.util import memoize
from .block_poutine import BlockPoutine
from .poutine import Poutine  # noqa: F401
from .queue_poutine import QueuePoutine
from .replay_poutine import ReplayPoutine
from .trace import Trace  # noqa: F401
from .trace_poutine import TracePoutine
from .tracegraph_poutine import TraceGraphPoutine
from .condition_poutine import ConditionPoutine


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

    def _fn(*args, **kwargs):
        p = TracePoutine(fn)
        return p(*args, **kwargs)

    return _fn


def tracegraph(fn, graph_output=None):
    def _fn(*args, **kwargs):
        p = TraceGraphPoutine(fn)
        return p(*args, **kwargs)

    return _fn


def replay(fn, trace, sites=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace

    ret = replay(fn, trace, sites=some_sites)(*args, **kwargs)
    """

    def _fn(*args, **kwargs):
        p = ReplayPoutine(fn, trace, sites=sites)
        return p(*args, **kwargs)

    return _fn


def block(fn, hide=None, expose=None, hide_types=None, expose_types=None):
    """
    Given a callable that contains Pyro primitive calls,
    hide the primitive calls at sites

    ret = block(fn, hide=["a"], expose=["b"])(*args, **kwargs)

    Also expose()?
    """

    def _fn(*args, **kwargs):
        p = BlockPoutine(fn, hide=hide, expose=expose,
                         hide_types=hide_types, expose_types=expose_types)
        return p(*args, **kwargs)

    return _fn


def queue(fn, queue=None, max_tries=None):
    """
    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue
    """

    def _fn(*args, **kwargs):
        p = QueuePoutine(fn, queue=queue, max_tries=max_tries)
        return p(*args, **kwargs)

    return _fn


def condition(fn, data=None):
    """
    Given a stochastic function with some sample statements
    and a dictionary of observations at names,
    change the sample statements at those names into observes
    with those values
    """
    def _fn(*args, **kwargs):
        p = ConditionPoutine(fn, data=data)
        return p(*args, **kwargs)
    return _fn


#########################################
# Begin composite operations
#########################################

def cache(fn, sites=None):
    """
    Given a callable that contains Pyro primitive calls, and sites or a pivot,
    run the callable once to get a trace and then replay the callable
    using the sites or pivot

    An example of using the poutine API to implement new composite control operations
    """
    memoized_trace = memoize(block(trace(fn)))

    def _fn(*args, **kwargs):
        tr = memoized_trace(*args, **kwargs)
        p = replay(fn, tr, sites=sites)
        return p(*args, **kwargs)

    return _fn


def do(fn, data=None):
    """
    Given a stochastic functuon with some sample statements
    and a dictionary of values at names,
    set the return values of those sites equal to the values
    and hide them from the rest of the stack
    as if they were hard-coded to those values
    """
    if data is None:
        data = {}

    def _fn(*args, **kwargs):
        p = block(condition(fn, data=data), hide=list(data.keys()))
        return p(*args, **kwargs)
    return _fn
