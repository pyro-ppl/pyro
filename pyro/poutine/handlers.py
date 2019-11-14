"""
Poutine is a library of composable effect handlers for recording and modifying the
behavior of Pyro programs. These lower-level ingredients simplify the implementation
of new inference algorithms and behavior.

Handlers can be used as higher-order functions, decorators, or context managers
to modify the behavior of functions or blocks of code:

For example, consider the following Pyro program:

    >>> def model(x):
    ...     s = pyro.param("s", torch.tensor(0.5))
    ...     z = pyro.sample("z", dist.Normal(x, s))
    ...     return z ** 2

We can mark sample sites as observed using ``condition``,
which returns a callable with the same input and output signatures as ``model``:

    >>> conditioned_model = poutine.condition(model, data={"z": 1.0})

We can also use handlers as decorators:

    >>> @pyro.condition(data={"z": 1.0})
    ... def model(x):
    ...     s = pyro.param("s", torch.tensor(0.5))
    ...     z = pyro.sample("z", dist.Normal(x, s))
    ...     return z ** 2

Or as context managers:

    >>> with pyro.condition(data={"z": 1.0}):
    ...     s = pyro.param("s", torch.tensor(0.5))
    ...     z = pyro.sample("z", dist.Normal(0., s))
    ...     y = z ** 2

Handlers compose freely:

    >>> conditioned_model = poutine.condition(model, data={"z": 1.0})
    >>> traced_model = poutine.trace(conditioned_model)

Many inference algorithms or algorithmic components can be implemented
in just a few lines of code::

    guide_tr = poutine.trace(guide).get_trace(...)
    model_tr = poutine.trace(poutine.replay(conditioned_model, trace=guide_tr)).get_trace(...)
    monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()
"""

import functools

from pyro.poutine import util
from pyro.poutine.messenger import Messenger
from pyro.util import get_rng_state, set_rng_seed, set_rng_state

from .block_messenger import BlockMessenger
from .broadcast_messenger import BroadcastMessenger
from .condition_messenger import ConditionMessenger
from .enumerate_messenger import EnumerateMessenger
from .escape_messenger import EscapeMessenger
from .infer_config_messenger import InferConfigMessenger
from .lift_messenger import LiftMessenger
from .markov_messenger import MarkovMessenger
from .mask_messenger import MaskMessenger
from .plate_messenger import PlateMessenger  # noqa F403
from .replay_messenger import ReplayMessenger
from .runtime import NonlocalExit
from .scale_messenger import ScaleMessenger
from .trace_messenger import TraceMessenger
from .uncondition_messenger import UnconditionMessenger

############################################
# Begin primitive operations
############################################


def trace(fn=None, graph_type=None, param_only=None):
    """
    Return a handler that records the inputs and outputs of primitive calls
    and their dependencies.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    We can record its execution using ``trace``
    and use the resulting data structure to compute the log-joint probability
    of all of the sample sites in the execution or extract all parameters.

        >>> trace = trace(model).get_trace(0.0)
        >>> logp = trace.log_prob_sum()
        >>> params = [trace.nodes[name]["value"].unconstrained() for name in trace.param_nodes]

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param graph_type: string that specifies the kind of graph to construct
    :param param_only: if true, only records params and not samples
    :returns: stochastic function decorated with a :class:`~pyro.poutine.trace_messenger.TraceMessenger`
    """
    msngr = TraceMessenger(graph_type=graph_type, param_only=param_only)
    return msngr(fn) if fn is not None else msngr


def replay(fn=None, trace=None, params=None):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    ``replay`` makes ``sample`` statements behave as if they had sampled the values
    at the corresponding sites in the trace:

        >>> old_trace = trace(model).get_trace(1.0)
        >>> replayed_model = replay(model, trace=old_trace)
        >>> bool(replayed_model(0.0) == old_trace.nodes["_RETURN"]["value"])
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param trace: a :class:`~pyro.poutine.Trace` data structure to replay against
    :param params: dict of names of param sites and constrained values
        in fn to replay against
    :returns: a stochastic function decorated with a :class:`~pyro.poutine.replay_messenger.ReplayMessenger`
    """
    msngr = ReplayMessenger(trace=trace, params=params)
    return msngr(fn) if fn is not None else msngr


def lift(fn=None, prior=None):
    """
    Given a stochastic function with param calls and a prior distribution,
    create a stochastic function where all param calls are replaced by sampling from prior.
    Prior should be a callable or a dict of names to callables.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2
        >>> lifted_model = lift(model, prior={"s": dist.Exponential(0.3)})

    ``lift`` makes ``param`` statements behave like ``sample`` statements
    using the distributions in ``prior``.  In this example, site `s` will now behave
    as if it was replaced with ``s = pyro.sample("s", dist.Exponential(0.3))``:

        >>> tr = trace(lifted_model).get_trace(0.0)
        >>> tr.nodes["s"]["type"] == "sample"
        True
        >>> tr2 = trace(lifted_model).get_trace(0.0)
        >>> bool((tr2.nodes["s"]["value"] == tr.nodes["s"]["value"]).all())
        False

    :param fn: function whose parameters will be lifted to random values
    :param prior: prior function in the form of a Distribution or a dict of stochastic fns
    :returns: ``fn`` decorated with a :class:`~pyro.poutine.lift_messenger.LiftMessenger`
    """
    msngr = LiftMessenger(prior=prior)
    return msngr(fn) if fn is not None else msngr


def block(fn=None, hide_fn=None, expose_fn=None, hide=None, expose=None, hide_types=None, expose_types=None):
    """
    This handler selectively hides Pyro primitive sites from the outside world.
    Default behavior: block everything.

    A site is hidden if at least one of the following holds:

        0. ``hide_fn(msg) is True`` or ``(not expose_fn(msg)) is True``
        1. ``msg["name"] in hide``
        2. ``msg["type"] in hide_types``
        3. ``msg["name"] not in expose and msg["type"] not in expose_types``
        4. ``hide``, ``hide_types``, and ``expose_types`` are all ``None``

    For example, suppose the stochastic function fn has two sample sites "a" and "b".
    Then any effect outside of ``BlockMessenger(fn, hide=["a"])``
    will not be applied to site "a" and will only see site "b":

        >>> def fn():
        ...     a = pyro.sample("a", dist.Normal(0., 1.))
        ...     return pyro.sample("b", dist.Normal(a, 1.))
        >>> fn_inner = trace(fn)
        >>> fn_outer = trace(block(fn_inner, hide=["a"]))
        >>> trace_inner = fn_inner.get_trace()
        >>> trace_outer  = fn_outer.get_trace()
        >>> "a" in trace_inner
        True
        >>> "a" in trace_outer
        False
        >>> "b" in trace_inner
        True
        >>> "b" in trace_outer
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param hide_fn: function that takes a site and returns True to hide the site
        or False/None to expose it.  If specified, all other parameters are ignored.
        Only specify one of hide_fn or expose_fn, not both.
    :param expose_fn: function that takes a site and returns True to expose the site
        or False/None to hide it.  If specified, all other parameters are ignored.
        Only specify one of hide_fn or expose_fn, not both.
    :param hide: list of site names to hide
    :param expose: list of site names to be exposed while all others hidden
    :param hide_types: list of site types to be hidden
    :param expose_types: list of site types to be exposed while all others hidden
    :returns: stochastic function decorated with a :class:`~pyro.poutine.block_messenger.BlockMessenger`
    """
    msngr = BlockMessenger(hide_fn=hide_fn, expose_fn=expose_fn,
                           hide=hide, expose=expose,
                           hide_types=hide_types, expose_types=expose_types)
    return msngr(fn) if fn is not None else msngr


def broadcast(fn=None):
    """
    Automatically broadcasts the batch shape of the stochastic function
    at a sample site when inside a single or nested plate context.
    The existing `batch_shape` must be broadcastable with the size
    of the :class:`~pyro.plate` contexts installed in the
    `cond_indep_stack`.

    Notice how `model_automatic_broadcast` below automates expanding of
    distribution batch shapes. This makes it easy to modularize a
    Pyro model as the sub-components are agnostic of the wrapping
    :class:`~pyro.plate` contexts.

    >>> def model_broadcast_by_hand():
    ...     with IndepMessenger("batch", 100, dim=-2):
    ...         with IndepMessenger("components", 3, dim=-1):
    ...             sample = pyro.sample("sample", dist.Bernoulli(torch.ones(3) * 0.5)
    ...                                                .expand_by(100))
    ...             assert sample.shape == torch.Size((100, 3))
    ...     return sample

    >>> @poutine.broadcast
    ... def model_automatic_broadcast():
    ...     with IndepMessenger("batch", 100, dim=-2):
    ...         with IndepMessenger("components", 3, dim=-1):
    ...             sample = pyro.sample("sample", dist.Bernoulli(torch.tensor(0.5)))
    ...             assert sample.shape == torch.Size((100, 3))
    ...     return sample
    """
    msngr = BroadcastMessenger()
    return msngr(fn) if fn is not None else msngr


def escape(fn=None, escape_fn=None):
    """
    Given a callable that contains Pyro primitive calls,
    evaluate escape_fn on each site, and if the result is True,
    raise a :class:`~pyro.poutine.runtime.NonlocalExit` exception that stops execution
    and returns the offending site.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param escape_fn: function that takes a partial trace and a site,
        and returns a boolean value to decide whether to exit at that site
    :returns: stochastic function decorated with :class:`~pyro.poutine.escape_messenger.EscapeMessenger`
    """
    msngr = EscapeMessenger(escape_fn)
    return msngr(fn) if fn is not None else msngr


def condition(fn=None, data=None):
    """
    Given a stochastic function with some sample statements
    and a dictionary of observations at names,
    change the sample statements at those names into observes
    with those values.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    To observe a value for site `z`, we can write

        >>> conditioned_model = condition(model, data={"z": torch.tensor(1.)})

    This is equivalent to adding `obs=value` as a keyword argument
    to `pyro.sample("z", ...)` in `model`.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param data: a dict or a :class:`~pyro.poutine.Trace`
    :returns: stochastic function decorated with a :class:`~pyro.poutine.condition_messenger.ConditionMessenger`
    """
    msngr = ConditionMessenger(data=data)
    return msngr(fn) if fn is not None else msngr


def uncondition(fn=None):
    """
    Given a stochastic funtion with sample statements, conditioned on observed
    values at some sample statements, removes the conditioning so that all
    nodes are sampled from.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :returns: a stochastic function decorated with a :class: `~pyro.poutine.uncondition_messenger.UnconditionMessenger`
    """
    msngr = UnconditionMessenger()
    return msngr(fn) if fn is not None else msngr


def infer_config(fn=None, config_fn=None):
    """
    Given a callable that contains Pyro primitive calls
    and a callable taking a trace site and returning a dictionary,
    updates the value of the infer kwarg at a sample site to config_fn(site).

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param config_fn: a callable taking a site and returning an infer dict
    :returns: stochastic function decorated with :class:`~pyro.poutine.infer_config_messenger.InferConfigMessenger`
    """
    msngr = InferConfigMessenger(config_fn)
    return msngr(fn) if fn is not None else msngr


def scale(fn=None, scale=None):
    """
    Given a stochastic function with some sample statements and a positive
    scale factor, scale the score of all sample and observe sites in the
    function.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s), obs=1.0)
        ...     return z ** 2

    ``scale`` multiplicatively scales the log-probabilities of sample sites:

        >>> scaled_model = scale(model, scale=0.5)
        >>> scaled_tr = trace(scaled_model).get_trace(0.0)
        >>> unscaled_tr = trace(model).get_trace(0.0)
        >>> bool((scaled_tr.log_prob_sum() == 0.5 * unscaled_tr.log_prob_sum()).all())
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param scale: a positive scaling factor
    :returns: stochastic function decorated with a :class:`~pyro.poutine.scale_messenger.ScaleMessenger`
    """
    msngr = ScaleMessenger(scale=scale)
    # XXX temporary compatibility fix
    return msngr(fn) if callable(fn) else msngr


def mask(fn=None, mask=None):
    """
    Given a stochastic function with some batched sample statements and
    masking tensor, mask out some of the sample statements elementwise.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param torch.BoolTensor mask: a ``{0,1}``-valued masking tensor
        (1 includes a site, 0 excludes a site)
    :returns: stochastic function decorated with a :class:`~pyro.poutine.scale_messenger.MaskMessenger`
    """
    msngr = MaskMessenger(mask=mask)
    return msngr(fn) if fn is not None else msngr


def enum(fn=None, first_available_dim=None):
    """
    Enumerates in parallel over discrete sample sites marked
    ``infer={"enumerate": "parallel"}``.

    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer.
    """
    assert first_available_dim < 0, first_available_dim
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
    by using ``block``.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    To intervene with a value for site `z`, we can write

        >>> intervened_model = do(model, data={"z": torch.tensor(1.)})

    This is equivalent to replacing `z = pyro.sample("z", ...)` with `z = value`.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param data: a ``dict`` or a :class:`~pyro.poutine.Trace`
    :returns: stochastic function decorated with a :class:`~pyro.poutine.block_messenger.BlockMessenger`
      and :class:`pyro.poutine.condition_messenger.ConditionMessenger`
    """
    def wrapper(wrapped):
        return block(condition(wrapped, data=data), hide=list(data.keys()))
    return wrapper(fn) if fn is not None else wrapper


def queue(fn=None, queue=None, max_tries=None,
          extend_fn=None, escape_fn=None, num_samples=None):
    """
    Used in sequential enumeration over discrete variables.

    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param queue: a queue data structure like multiprocessing.Queue to hold partial traces
    :param max_tries: maximum number of attempts to compute a single complete trace
    :param extend_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a list of extended traces
    :param escape_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a boolean value to decide whether to exit
    :param num_samples: optional number of extended traces for extend_fn to return
    :returns: stochastic function decorated with poutine logic
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

            for i in range(max_tries):
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


def markov(fn=None, history=1, keep=False):
    """
    Markov dependency declaration.

    This can be used in a variety of ways:
    - as a context manager
    - as a decorator for recursive functions
    - as an iterator for markov chains

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their share"
    """
    if fn is None:
        # Used as a decorator with bound args
        return MarkovMessenger(history=history, keep=keep)
    if not callable(fn):
        # Used as a generator
        return MarkovMessenger(history=history, keep=keep).generator(iterable=fn)
    # Used as a decorator with bound args
    return MarkovMessenger(history=history, keep=keep)(fn)


class _SeedMessenger(Messenger):
    def __init__(self, rng_seed):
        assert isinstance(rng_seed, int)
        self.rng_seed = rng_seed
        super(_SeedMessenger, self).__init__()

    def __enter__(self):
        self.old_state = get_rng_state()
        set_rng_seed(self.rng_seed)

    def __exit__(self, type, value, traceback):
        set_rng_state(self.old_state)


def seed(fn=None, rng_seed=None):
    """
    Handler to set the random number generator to a pre-defined state by setting its
    seed. This is the same as calling :func:`pyro.set_rng_seed` before the
    call to `fn`. This handler has no additional effect on primitive statements on the
    standard Pyro backend, but it might intercept ``pyro.sample`` calls in other
    backends. e.g. the NumPy backend.

    :param fn: a stochastic function (callable containing Pyro primitive calls).
    :param int rng_seed: rng seed.
    """
    msngr = _SeedMessenger(rng_seed)
    return msngr(fn) if fn is not None else msngr
