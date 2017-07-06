import pyro
from pyro.util import memoize

from .poutine import Poutine
from .block_poutine import BlockPoutine
from .trace_poutine import TracePoutine
from .replay_poutine import ReplayPoutine


def record(fn):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that records the inputs and outputs to those primitive calls
    
    Adds trace data structure site constructors to primitive stacks
    
    trace = record(fn)(*args, **kwargs)
    """
    pass


def replay(fn, trace, sites=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace
    
    ret = replay(fn, trace=some_trace, sites=some_sites)(*args, **kwargs)
    """
    replayer = ReplayPoutine(fn, trace)
    pass


def suppress(fn, hide=None, expose=None):
    """
    Given a callable that contains Pyro primitive calls,
    hide the primitive calls at sites
    
    ret = suppress(fn, include=["a"], exclude=["b"])(*args, **kwargs)
    
    Also expose()?
    """
    pass


def cache(fn, sites, *args, **kwargs):
    """
    An example of using the API?
    """
    memoized_record = memoize(record(fn))
    def cached(*args, **kwargs):
        tr, y = memoized_record(*args, **kwargs)
        return replay(fn, trace=tr, sites=sites)
    return cached

