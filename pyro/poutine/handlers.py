from __future__ import absolute_import, division, print_function

import functools

from six.moves import xrange

from pyro.poutine import util

from .block_messenger import BlockMessenger
from .condition_messenger import ConditionMessenger
from .enumerate_messenger import EnumerateMessenger
from .escape_messenger import EscapeMessenger
from .indep_messenger import IndepMessenger
from .infer_config_messenger import InferConfigMessenger
from .lift_messenger import LiftMessenger
from .replay_messenger import ReplayMessenger
from .runtime import NonlocalExit
from .scale_messenger import ScaleMessenger
from .trace_messenger import TraceMessenger

############################################
# Begin primitive operations
############################################


def trace(fn=None, graph_type=None, param_only=None):
    """
    Return a handler that records the inputs and outputs of primitive calls
    and their dependencies.

    Adds trace data structure site constructors to primitive stacks

    Alias for :class:`TraceMessenger` constructor.

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param graph_type: string that specifies the kind of graph to construct
    :param param_only: if true, only records params and not samples
    :returns: stochastic function wrapped in a TraceHandler
    :rtype: TraceHandler
    """
    msngr = TraceMessenger(graph_type=graph_type, param_only=param_only)
    return msngr(fn) if fn is not None else msngr


def replay(fn=None, trace=None, sites=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param trace: a Trace data structure to replay against
    :param sites: list or dict of names of sample sites in fn to replay against,
        defaulting to all sites
    :returns: a replay messenger
    :rtype: ReplayMessenger

    """
    msngr = ReplayMessenger(trace=trace, sites=sites)
    return msngr(fn) if fn is not None else msngr


def lift(fn=None, prior=None):
    """
    Given a stochastic function with param calls and a prior distribution,
    create a stochastic function where all param calls are replaced by sampling from prior.
    Prior should be a callable or a dict of names to callables.

    :param fn: function whose parameters will be lifted to random values
    :param prior: prior function in the form of a Distribution or a dict of stochastic fns
    :returns: stochastic function wrapped in LiftHandler
    """
    msngr = LiftMessenger(prior=prior)
    return msngr(fn) if fn is not None else msngr


def block(fn=None, hide=None, expose=None, hide_types=None, expose_types=None):
    """
    Given a callable that contains Pyro primitive calls,
    selectively hide some of those calls from poutines higher up the stack

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param hide: list of site names to hide
    :param expose: list of site names to be exposed while all others hidden
    :param hide_types: list of site types to be hidden
    :param expose_types: list of site types to be exposed while all others hidden
    :returns: stochastic function wrapped in a BlockHandler
    :rtype: BlockHandler
    """
    msngr = BlockMessenger(hide=hide, expose=expose,
                           hide_types=hide_types, expose_types=expose_types)
    return msngr(fn) if fn is not None else msngr


def escape(fn=None, escape_fn=None):
    """
    Given a callable that contains Pyro primitive calls,
    evaluate escape_fn on each site, and if the result is True,
    raise a NonlocalExit exception that stops execution
    and returns the offending site.

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param escape_fn: function that takes a partial trace and a site,
        and returns a boolean value to decide whether to exit at that site
    :returns: stochastic function wrapped in EscapeHandler
    """
    msngr = EscapeMessenger(escape_fn)
    return msngr(fn) if fn is not None else msngr


def condition(fn=None, data=None):
    """
    Given a stochastic function with some sample statements
    and a dictionary of observations at names,
    change the sample statements at those names into observes
    with those values

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param data: a dict or a Trace
    :returns: stochastic function wrapped in a ConditionHandler
    :rtype: ConditionHandler
    """
    msngr = ConditionMessenger(data=data)
    return msngr(fn) if fn is not None else msngr


def infer_config(fn=None, config_fn=None):
    """
    Given a callable that contains Pyro primitive calls
    and a callable taking a trace site and returning a dictionary,
    updates the value of the infer kwarg at a sample site to config_fn(site)

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param config_fn: a callable taking a site and returning an infer dict
    :returns: stochastic function wrapped in InferConfigHandler
    """
    msngr = InferConfigMessenger(config_fn)
    return msngr(fn) if fn is not None else msngr


def scale(fn=None, scale=None):
    """
    Given a stochastic function with some sample statements and a positive
    scale factor, scale the score of all sample and observe sites in the
    function.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param scale: a positive scaling factor
    :returns: stochastic function wrapped in a ScaleHandler
    :rtype: ScaleMessenger
    """
    msngr = ScaleMessenger(scale=scale)
    # XXX temporary compatibility fix
    return msngr(fn) if callable(fn) else msngr


def indep(fn=None, name=None, size=None, dim=None):
    """
    Note: Low-level; use `pyro.iarange` instead.

    This messenger keeps track of stack of independence information declared by
    nested ``irange`` and ``iarange`` contexts. This information is stored in
    a ``cond_indep_stack`` at each sample/observe site for consumption by
    ``TraceMessenger``.
    """
    msngr = IndepMessenger(name=name, size=size, dim=dim)
    return msngr(fn) if fn is not None else msngr


def enum(fn=None, first_available_dim=None):
    """
    Enumerates in parallel over discrete sample sites marked
    ``infer={"enumerate": "parallel"}``.

    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
    """
    msngr = EnumerateMessenger(first_available_dim=first_available_dim)
    return msngr(fn) if fn is not None else msngr


#########################################
# Begin composite operations
#########################################

def do(fn=None, data=None):
    """
    Given a stochastic function with some sample statements
    and a dictionary of values at names,
    set the return values of those sites equal to the values
    and hide them from the rest of the stack
    as if they were hard-coded to those values
    by using BlockHandler

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param data: a dict or a Trace
    :returns: stochastic function wrapped in a BlockHandler and ConditionHandler
    """
    def wrapper(wrapped):
        return block(condition(wrapped, data=data), hide=list(data.keys()))
    return wrapper(fn) if fn is not None else wrapper


def queue(fn=None, queue=None, max_tries=None,
          extend_fn=None, escape_fn=None, num_samples=None):
    """
    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue

    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param queue: a queue data structure like multiprocessing.Queue to hold partial traces
    :param max_tries: maximum number of attempts to compute a single complete trace
    :param extend_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a list of extended traces
    :param escape_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a boolean value to decide whether to exit
    :param num_samples: optional number of extended traces for extend_fn to return
    :returns: stochastic function wrapped in poutine logic
    """

    if max_tries is None:
        max_tries = int(1e6)

    if extend_fn is None:
        extend_fn = util.enum_extend

    if escape_fn is None:
        escape_fn = util.discrete_escape

    if num_samples is None:
        num_samples = -1

    def wrapper(wrapped):
        def _fn(*args, **kwargs):

            for i in xrange(max_tries):
                assert not queue.empty(), \
                    "trying to get() from an empty queue will deadlock"

                next_trace = queue.get()
                try:
                    ftr = trace(escape(replay(wrapped, trace=next_trace),
                                       escape_fn=functools.partial(escape_fn,
                                                                   next_trace)))
                    return ftr(*args, **kwargs)
                except NonlocalExit as site_container:
                    site_container.reset_stack()
                    for tr in extend_fn(ftr.trace.copy(), site_container.site,
                                        num_samples=num_samples):
                        queue.put(tr)

            raise ValueError("max tries ({}) exceeded".format(str(max_tries)))
        return _fn

    return wrapper(fn) if fn is not None else wrapper
