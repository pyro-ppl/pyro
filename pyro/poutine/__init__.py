import functools
from six.moves.queue import LifoQueue

# poutines
from .block_poutine import BlockPoutine
from .poutine import Poutine  # noqa: F401
from .branch_poutine import BranchPoutine
from .replay_poutine import ReplayPoutine
from .trace_poutine import TracePoutine
from .tracegraph_poutine import TraceGraphPoutine
from .lift_poutine import LiftPoutine
from .condition_poutine import ConditionPoutine
from .lambda_poutine import LambdaPoutine  # noqa: F401
from .escape_poutine import EscapePoutine

# trace data structures
from .trace import Trace, TraceGraph  # noqa: F401
from pyro import util


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
                for tr in extend_fn(ftr.trace.copy(), site_container.site,
                                    num_samples=num_samples):
                    queue.put(tr)

        raise ValueError("max tries ({}) exceeded".format(str(max_tries)))

    return _fn


def iter_discrete(fn):
    """
    Iterate over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this iterates over all choices
    and scales entire traces by the discrete choice probability.

    :param callable fn: A stochastic function.
    :returns: An iterator over results of `fn`.
    """

    @functools.wraps(fn)
    def iter_fn(*args, **kwargs):
        queue = LifoQueue()
        queue.put(Trace())
        while not queue.empty():
            partial_trace = queue.get()
            escape_fn = functools.partial(util.discrete_escape, partial_trace)
            traced_fn = trace(escape(replay(fn, partial_trace), escape_fn))
            try:
                yield traced_fn(*args, **kwargs)
            except util.NonlocalExit as e:
                for tr in util.enum_extend(traced_fn.trace.copy(), e.site):
                    # TODO Scale traces by the choice probability.
                    queue.put(tr)

    return iter_fn


def branch_discrete(fn):
    """
    Batch over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this batches over all choices
    and scales entire traces by the discrete choice probability.

    Note that this changes the tensor shapes of the resulting program by
    adding tensor dimensions on the left. To write programs that work
    correctly with `branch_discrete`, simply index tensors from the right.

    :param callable fn: A stochastic function.
    :returns: A stochastic function wrapped in a BranchPoutine.
    :rtype: pyro.poutine.BranchPoutine

    Examples:

        >>> def gaussian_mixture_model():
                ps = pyro.get_param("ps", Variable(torch.ones(10) / 10))
                mus = pyro.get_param("mus", Variable(torch.zeros(10))
                sigma = pyro.get_param("sigma", Variable(torch.ones(1))
                z = pyro.sample("z", dist.categorical, ps, one_hot=False)
                x = pyro.samples("x", dist.diagnormal, mus[z], sigma)
                return x
        >>> gaussian_mixture_model().size()
        (1L,)
        >>> branch_discrete(gaussian_mixture_model)().size()
        (10L, 1L)
    """
    return BranchPoutine(fn)
