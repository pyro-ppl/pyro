from __future__ import absolute_import, division, print_function

import functools

from pyro import util

# poutines
from .block_poutine import BlockPoutine
from .condition_poutine import ConditionPoutine
from .escape_poutine import EscapePoutine
from .indep_poutine import IndepPoutine  # noqa: F401
from .lift_poutine import LiftPoutine
from .poutine import _PYRO_STACK, Poutine  # noqa: F401
from .replay_poutine import ReplayPoutine
from .scale_poutine import ScalePoutine
from .trace import Trace  # noqa: F401
from .trace_poutine import TracePoutine

############################################
# Begin primitive operations
############################################


def trace(fn, graph_type=None):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param graph_type: string that specifies the kind of graph to construct
    :returns: stochastic function wrapped in a TracePoutine
    :rtype: pyro.poutine.TracePoutine

    Alias for TracePoutine constructor.

    Given a callable that contains Pyro primitive calls, return a TracePoutine callable
    that records the inputs and outputs to those primitive calls
    and their dependencies.

    Adds trace data structure site constructors to primitive stacks
    """
    return TracePoutine(fn, graph_type=graph_type)


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


def lift(fn, prior):
    """
    :param fn: function whose parameters will be lifted to random values
    :param prior: prior function in the form of a Distribution or a dict of stochastic fns
    :returns: stochastic function wrapped in LiftPoutine

    Given a stochastic function with param calls and a prior distribution,
    create a stochastic function where all param calls are replaced by sampling from prior.
    Prior should be a callable or a dict of names to callables.
    """
    return LiftPoutine(fn, prior)


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


def escape(fn, escape_fn=None):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param escape_fn: function that takes a partial trace and a site
    and returns a boolean value to decide whether to exit at that site
    :returns: stochastic function wrapped in EscapePoutine

    Alias for EscapePoutine constructor.

    Given a callable that contains Pyro primitive calls,
    evaluate escape_fn on each site, and if the result is True,
    raise a NonlocalExit exception that stops execution
    and returns the offending site.
    """
    return EscapePoutine(fn, escape_fn)


def condition(fn, data):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param data: a dict or a Trace
    :returns: stochastic function wrapped in a ConditionPoutine
    :rtype: pyro.poutine.ConditionPoutine

    Alias for ConditionPoutine constructor.

    Given a stochastic function with some sample statements
    and a dictionary of observations at names,
    change the sample statements at those names into observes
    with those values
    """
    return ConditionPoutine(fn, data=data)


def indep(fn, name, vectorized):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param str name: a name for subsample sites
    :param bool vectorized: True for ``iarange``, False for ``irange``
    :returns: stochastic function wrapped in an IndepPoutine
    :rtype: pyro.poutine.IndepPoutine

    Alias for IndepPoutine constructor.

    Used internally by ``iarange`` and ``irange``.
    """
    return IndepPoutine(fn, name=name, vectorized=vectorized)


def scale(fn, scale):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param scale: a positive scaling factor
    :returns: stochastic function wrapped in a ScalePoutine
    :rtype: pyro.poutine.ScalePoutine

    Alias for ScalePoutine constructor.

    Given a stochastic function with some sample statements and a positive
    scale factor, scale the score of all sample and observe sites in the
    function.
    """
    return ScalePoutine(fn, scale=scale)


#########################################
# Begin composite operations
#########################################

def do(fn, data):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param data: a dict or a Trace
    :returns: stochastic function wrapped in a BlockPoutine and ConditionPoutine
    :rtype: pyro.poutine.BlockPoutine

    Given a stochastic function with some sample statements
    and a dictionary of values at names,
    set the return values of those sites equal to the values
    and hide them from the rest of the stack
    as if they were hard-coded to those values
    by using BlockPoutine
    """
    return BlockPoutine(ConditionPoutine(fn, data=data),
                        hide=list(data.keys()))


def queue(fn, queue, max_tries=None,
          extend_fn=None, escape_fn=None, num_samples=None):
    """
    :param fn: a stochastic function (callable containing pyro primitive calls)
    :param queue: a queue data structure like multiprocessing.Queue to hold partial traces
    :param max_tries: maximum number of attempts to compute a single complete trace
    :param extend_fn: function (possibly stochastic) that takes a partial trace and a site
    and returns a list of extended traces
    :param escape_fn: function (possibly stochastic) that takes a partial trace and a site
    and returns a boolean value to decide whether to exit
    :param num_samples: optional number of extended traces for extend_fn to return
    :returns: stochastic function wrapped in poutine logic

    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue
    """

    if max_tries is None:
        max_tries = int(1e6)

    if extend_fn is None:
        # XXX should be util.enum_extend
        extend_fn = util.enum_extend

    if escape_fn is None:
        # XXX should be util.discrete_escape
        escape_fn = util.discrete_escape

    if num_samples is None:
        num_samples = -1

    def _fn(*args, **kwargs):

        for i in range(max_tries):
            assert not queue.empty(), \
                "trying to get() from an empty queue will deadlock"

            next_trace = queue.get()
            try:
                ftr = trace(escape(replay(fn, next_trace),
                                   functools.partial(escape_fn, next_trace)))
                return ftr(*args, **kwargs)
            except util.NonlocalExit as site_container:
                for frame in _PYRO_STACK:
                    frame._reset()
                for tr in extend_fn(ftr.trace.copy(), site_container.site,
                                    num_samples=num_samples):
                    queue.put(tr)

        raise ValueError("max tries ({}) exceeded".format(str(max_tries)))

    return _fn
