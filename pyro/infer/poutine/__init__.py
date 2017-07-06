import pyro
from pyro.util import memoize

from .poutine import Poutine
from .block_poutine import BlockPoutine
from .trace_poutine import TracePoutine
from .replay_poutine import ReplayPoutine
from .pivot_poutine import PivotPoutine
from .beam_poutine import BeamPoutine


############################################
# Begin primitive operations
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


def pivot(fn, trace, pivot=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    up until the pivot site is reached, then draws new samples from the model
    """
    def _fn(*args, **kwargs):
        p = PivotPoutine(fn, trace, pivot=pivot)
        return p(*args, **kwargs)
    return _fn


def block(fn, hide=None, expose=None):
    """
    Given a callable that contains Pyro primitive calls,
    hide the primitive calls at sites
    
    ret = block(fn, hide=["a"], expose=["b"])(*args, **kwargs)
    
    Also expose()?
    """
    def _fn(*args, **kwargs):
        p = BlockPoutine(fn, hide=hide, expose=expose)
        return p(*args, **kwargs)
    return _fn


def beam(fn, queue=None, max_tries=None):
    """
    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue
    """
    def _fn(*args, **kwargs):
        p = BeamPoutine(fn, queue=queue, max_tries=max_tries)
        return p(*args, **kwargs)
    return _fn


#########################################
# Begin composite operations
#########################################

def cache(fn, sites=None, pivot=None):
    """
    Given a callable that contains Pyro primitive calls, and sites or a pivot,
    run the callable once to get a trace and then replay the callable
    using the sites or pivot
    
    An example of using the poutine API to implement new composite control operations
    """
    assert(sites is None or pivot is None, "only provide one replay type")
    memoized_trace = memoize(trace(fn))
    def _fn(*args, **kwargs):
        tr = memoized_trace(*args, **kwargs)
        if sites is not None:
            p = replay(fn, trace=tr, sites=sites)
        else:
            p = pivot(fn, trace=tr, pivot=pivot)
        return p(*args, **kwargs)
    return _fn


def collapse(fn):
    """
    Given a callable that contains Pyro primitive calls and returns Traces,
    return a callable that returns the trace's return value
    """
    def _fn(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, pyro.infer.Trace):
            return ret["_RETURN"]["value"]
        else:
            return ret
    return _fn


def inject(model, guide, sites=None, pivot=None):
    """
    Given two stochastic functions and some sites or a pivot,
    get a trace from the guide function and replay the model from it
    using the sites or pivot
    """
    assert(sites is None or pivot is None, "only provide one replay type")
    def _fn(*args, **kwargs):
        guide_trace = trace(guide)(*args, **kwargs)
        if sites is not None:
            p = replay(model, guide_trace, sites=sites)
        else:
            p = pivot(model, guide_trace, pivot=pivot)
        return p(*args, **kwargs)
    return _fn
