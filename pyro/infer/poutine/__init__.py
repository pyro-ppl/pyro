import pyro
from pyro.util import memoize

from .poutine import Poutine
from .block_poutine import BlockPoutine
from .trace_poutine import TracePoutine
from .replay_poutine import ReplayPoutine
from .pivot_poutine import PivotPoutine


def trace(fn):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that records the inputs and outputs to those primitive calls
    
    Adds trace data structure site constructors to primitive stacks
    
    trace = record(fn)(*args, **kwargs)
    """
    def _fn(*args, **kwargs):
        p = TracePoutine(fn)
        return p(*args, **kwargs)
    return _fn


def replay(fn, trace, sites=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace
    
    ret = replay(fn, trace=some_trace, sites=some_sites)(*args, **kwargs)
    """
    def _fn(*args, **kwargs):
        p = ReplayPoutine(fn, trace, sites=sites)
        return p(*args, **kwargs)
    return _fn


def pivot(fn, trace, pivot=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    up until the pivot site is reached
    """
    def _fn(*args, **kwargs):
        p = PivotPoutine(fn, trace, pivot=pivot)
        return p(*args, **kwargs)
    return _fn


def block(fn, hide=None, expose=None):
    """
    Given a callable that contains Pyro primitive calls,
    hide the primitive calls at sites
    
    ret = suppress(fn, include=["a"], exclude=["b"])(*args, **kwargs)
    
    Also expose()?
    """
    def _fn(*args, **kwargs):
        p = BlockPoutine(fn, hide=hide, expose=expose)
        return p(*args, **kwargs)
    return _fn


def cache(fn, sites=None, pivot=None):
    """
    An example of using the API?
    """
    memoized_trace = memoize(trace(fn))
    def _fn(*args, **kwargs):
        tr = memoized_trace(*args, **kwargs)
        if sites is not None:
            p = replay(fn, trace=tr, sites=sites)
        else:
            p = pivot(fn, trace=tr, pivot=pivot)
        return p(*args, **kwargs)
    return _fn

