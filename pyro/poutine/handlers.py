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
    model_tr = poutine.trace(poutine.replay(conditioned_model, trace=tr)).get_trace(...)
    monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()
"""

from __future__ import absolute_import, division, print_function

import functools
import types

from six.moves import xrange

from pyro.poutine import util

from .broadcast_messenger import BroadcastMessenger
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


class Handler(object):

    def __init__(self, messenger):
        self._impl = messenger

    def __call__(self, fn=None, *args, **kwargs):
        if isinstance(self._impl, types.FunctionType):
            msngr = lambda _fn: self._impl(_fn, *args, **kwargs)  # noqa: E731
        else:
            msngr = self._impl(*args, **kwargs)
        return msngr(fn) if fn is not None else msngr

    @property
    def __doc__(self):
        return getattr(self._impl, "__doc__", None)

    def set(self, messenger):
        self._impl = messenger


broadcast = Handler(messenger=BroadcastMessenger)
block = Handler(messenger=BlockMessenger)
condition = Handler(messenger=ConditionMessenger)
enum = Handler(messenger=EnumerateMessenger)
escape = Handler(messenger=EscapeMessenger)
indep = Handler(messenger=IndepMessenger)
infer_config = Handler(messenger=InferConfigMessenger)
lift = Handler(messenger=LiftMessenger)
replay = Handler(messenger=ReplayMessenger)
scale = Handler(messenger=ScaleMessenger)
trace = Handler(messenger=TraceMessenger)


@Handler
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
    return block(condition(fn, data=data), hide=list(data.keys()))


@Handler
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

    def _fn(*args, **kwargs):

        for i in xrange(max_tries):
            assert not queue.empty(), \
                "trying to get() from an empty queue will deadlock"

            next_trace = queue.get()
            try:
                ftr = trace(escape(replay(fn, trace=next_trace),
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
