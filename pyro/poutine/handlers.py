# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

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

import collections
import functools
from typing import Callable, Iterable, Optional, TypeVar, Union, overload

from typing_extensions import ParamSpec

from pyro.poutine import util

from .block_messenger import BlockMessenger
from .broadcast_messenger import BroadcastMessenger
from .collapse_messenger import CollapseMessenger
from .condition_messenger import ConditionMessenger
from .do_messenger import DoMessenger
from .enum_messenger import EnumMessenger
from .escape_messenger import EscapeMessenger
from .infer_config_messenger import InferConfigMessenger
from .lift_messenger import LiftMessenger
from .markov_messenger import MarkovMessenger
from .mask_messenger import MaskMessenger
from .reparam_messenger import ReparamMessenger
from .replay_messenger import ReplayMessenger
from .runtime import NonlocalExit
from .scale_messenger import ScaleMessenger
from .seed_messenger import SeedMessenger
from .substitute_messenger import SubstituteMessenger
from .trace_messenger import TraceMessenger
from .uncondition_messenger import UnconditionMessenger

_P = ParamSpec("_P")
_T = TypeVar("_T")

############################################
# Begin primitive operations
############################################


def _make_handler(msngr_cls, module=None):
    def handler_decorator(func):
        @functools.wraps(func)
        def handler(fn=None, *args, **kwargs):
            if fn is not None and not (
                callable(fn) or isinstance(fn, collections.abc.Iterable)
            ):
                raise ValueError(
                    f"{fn} is not callable, did you mean to pass it as a keyword arg?"
                )
            msngr = msngr_cls(*args, **kwargs)
            return (
                functools.update_wrapper(msngr(fn), fn, updated=())
                if fn is not None
                else msngr
            )

        handler.__doc__ = (
            """Convenient wrapper of :class:`~pyro.poutine.{}.{}` \n\n""".format(
                func.__name__ + "_messenger", msngr_cls.__name__
            )
            + (msngr_cls.__doc__ if msngr_cls.__doc__ else "")
        )
        if module is not None:
            handler.__module__ = module
        return handler

    return handler_decorator


@_make_handler(BlockMessenger)
def block(
    fn=None,
    hide_fn=None,
    expose_fn=None,
    hide_all=True,
    expose_all=False,
    hide=None,
    expose=None,
    hide_types=None,
    expose_types=None,
): ...


@_make_handler(BroadcastMessenger)
def broadcast(fn=None): ...


@_make_handler(CollapseMessenger)
def collapse(fn=None, *args, **kwargs): ...


@_make_handler(ConditionMessenger)
def condition(fn, data): ...


@_make_handler(DoMessenger)
def do(fn, data): ...


@_make_handler(EnumMessenger)
def enum(fn=None, first_available_dim=None): ...


@_make_handler(EscapeMessenger)
def escape(fn, escape_fn): ...


@_make_handler(InferConfigMessenger)
def infer_config(fn, config_fn): ...


@_make_handler(LiftMessenger)
def lift(fn, prior): ...


@_make_handler(MaskMessenger)
def mask(fn, mask): ...


@_make_handler(ReparamMessenger)
def reparam(fn, config): ...


@_make_handler(ReplayMessenger)
def replay(fn=None, trace=None, params=None): ...


@_make_handler(ScaleMessenger)
def scale(fn, scale): ...


@_make_handler(SeedMessenger)
def seed(fn, rng_seed): ...


@_make_handler(TraceMessenger)
def trace(fn=None, graph_type=None, param_only=None): ...


@_make_handler(UnconditionMessenger)
def uncondition(fn=None): ...


@_make_handler(SubstituteMessenger)
def substitute(fn, data): ...


#########################################
# Begin composite operations
#########################################


def queue(
    fn=None,
    queue=None,
    max_tries=None,
    extend_fn=None,
    escape_fn=None,
    num_samples=None,
):
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
                assert (
                    not queue.empty()
                ), "trying to get() from an empty queue will deadlock"

                next_trace = queue.get()
                try:
                    ftr = trace(
                        escape(
                            replay(wrapped, trace=next_trace),  # noqa: F821
                            escape_fn=functools.partial(escape_fn, next_trace),
                        )
                    )
                    return ftr(*args, **kwargs)
                except NonlocalExit as site_container:
                    site_container.reset_stack()
                    for tr in extend_fn(
                        ftr.trace.copy(), site_container.site, num_samples=num_samples
                    ):
                        queue.put(tr)

            raise ValueError("max tries ({}) exceeded".format(str(max_tries)))

        return _fn

    return wrapper(fn) if fn is not None else wrapper


@overload
def markov(
    fn: None = ...,
    history: int = 1,
    keep: bool = False,
    dim: Optional[int] = None,
    name: Optional[str] = None,
) -> MarkovMessenger: ...


@overload
def markov(
    fn: Iterable[int] = ...,
    history: int = 1,
    keep: bool = False,
    dim: Optional[int] = None,
    name: Optional[str] = None,
) -> MarkovMessenger: ...


@overload
def markov(
    fn: Callable[_P, _T] = ...,
    history: int = 1,
    keep: bool = False,
    dim: Optional[int] = None,
    name: Optional[str] = None,
) -> Callable[_P, _T]: ...


def markov(
    fn: Optional[Union[Iterable[int], Callable]] = None,
    history: int = 1,
    keep: bool = False,
    dim: Optional[int] = None,
    name: Optional[str] = None,
) -> Union[MarkovMessenger, Callable]:
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
    :param int dim: An optional dimension to use for this independence index.
        Interface stub, behavior not yet implemented.
    :param str name: An optional unique name to help inference algorithms match
        :func:`pyro.markov` sites between models and guides.
        Interface stub, behavior not yet implemented.
    """
    if fn is None:
        # Used as a decorator with bound args
        return MarkovMessenger(history=history, keep=keep, dim=dim, name=name)
    if not callable(fn):
        # Used as a generator
        return MarkovMessenger(
            history=history, keep=keep, dim=dim, name=name
        ).generator(iterable=fn)
    # Used as a decorator with bound args
    return MarkovMessenger(history=history, keep=keep, dim=dim, name=name)(fn)
