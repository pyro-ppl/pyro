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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import ParamSpec

from pyro.poutine import util
from pyro.poutine.block_messenger import BlockMessenger
from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.collapse_messenger import CollapseMessenger
from pyro.poutine.condition_messenger import ConditionMessenger
from pyro.poutine.do_messenger import DoMessenger
from pyro.poutine.enum_messenger import EnumMessenger
from pyro.poutine.equalize_messenger import EqualizeMessenger
from pyro.poutine.escape_messenger import EscapeMessenger
from pyro.poutine.infer_config_messenger import InferConfigMessenger
from pyro.poutine.lift_messenger import LiftMessenger
from pyro.poutine.markov_messenger import MarkovMessenger
from pyro.poutine.mask_messenger import MaskMessenger
from pyro.poutine.reparam_messenger import ReparamHandler, ReparamMessenger
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.runtime import NonlocalExit
from pyro.poutine.scale_messenger import ScaleMessenger
from pyro.poutine.seed_messenger import SeedMessenger
from pyro.poutine.substitute_messenger import SubstituteMessenger
from pyro.poutine.trace_messenger import TraceHandler, TraceMessenger
from pyro.poutine.uncondition_messenger import UnconditionMessenger

if TYPE_CHECKING:
    import numbers

    import torch

    from pyro.distributions.distribution import Distribution
    from pyro.infer.reparam.reparam import Reparam
    from pyro.poutine.runtime import InferDict, Message
    from pyro.poutine.trace_struct import Trace

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


@overload
def block(
    fn: None = ...,
    hide_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    expose_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    hide_all: bool = True,
    expose_all: bool = False,
    hide: Optional[List[str]] = None,
    expose: Optional[List[str]] = None,
    hide_types: Optional[List[str]] = None,
    expose_types: Optional[List[str]] = None,
) -> BlockMessenger: ...


@overload
def block(
    fn: Callable[_P, _T],
    hide_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    expose_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    hide_all: bool = True,
    expose_all: bool = False,
    hide: Optional[List[str]] = None,
    expose: Optional[List[str]] = None,
    hide_types: Optional[List[str]] = None,
    expose_types: Optional[List[str]] = None,
) -> Callable[_P, _T]: ...


@_make_handler(BlockMessenger)
def block(  # type: ignore[empty-body]
    fn: Optional[Callable[_P, _T]] = None,
    hide_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    expose_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    hide_all: bool = True,
    expose_all: bool = False,
    hide: Optional[List[str]] = None,
    expose: Optional[List[str]] = None,
    hide_types: Optional[List[str]] = None,
    expose_types: Optional[List[str]] = None,
) -> Union[BlockMessenger, Callable[_P, _T]]: ...


@overload
def broadcast(
    fn: None = ...,
) -> BroadcastMessenger: ...


@overload
def broadcast(
    fn: Callable[_P, _T],
) -> Callable[_P, _T]: ...


@_make_handler(BroadcastMessenger)
def broadcast(  # type: ignore[empty-body]
    fn: Optional[Callable[_P, _T]] = None,
) -> Union[BroadcastMessenger, Callable[_P, _T]]: ...


@overload
def collapse(
    fn: None = ...,
    *args: Any,
    **kwargs: Any,
) -> CollapseMessenger: ...


@overload
def collapse(
    fn: Callable[_P, _T],
    *args: Any,
    **kwargs: Any,
) -> Callable[_P, _T]: ...


@_make_handler(CollapseMessenger)
def collapse(  # type: ignore[empty-body]
    fn: Optional[Callable[_P, _T]] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[CollapseMessenger, Callable[_P, _T]]: ...


@overload
def condition(
    data: Union[Dict[str, "torch.Tensor"], "Trace"],
) -> ConditionMessenger: ...


@overload
def condition(
    fn: Callable[_P, _T],
    data: Union[Dict[str, "torch.Tensor"], "Trace"],
) -> Callable[_P, _T]: ...


@_make_handler(ConditionMessenger)
def condition(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    data: Union[Dict[str, "torch.Tensor"], "Trace"],
) -> Union[ConditionMessenger, Callable[_P, _T]]: ...


@overload
def do(
    data: Dict[str, Union["torch.Tensor", "numbers.Number"]],
) -> DoMessenger: ...


@overload
def do(
    fn: Callable[_P, _T],
    data: Dict[str, Union["torch.Tensor", "numbers.Number"]],
) -> Callable[_P, _T]: ...


@_make_handler(DoMessenger)
def do(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    data: Dict[str, Union["torch.Tensor", "numbers.Number"]],
) -> Union[DoMessenger, Callable[_P, _T]]: ...


@overload
def enum(
    fn: None = ...,
    first_available_dim: Optional[int] = None,
) -> EnumMessenger: ...


@overload
def enum(
    fn: Callable[_P, _T],
    first_available_dim: Optional[int] = None,
) -> Callable[_P, _T]: ...


@_make_handler(EnumMessenger)
def enum(  # type: ignore[empty-body]
    fn: Optional[Callable[_P, _T]] = None,
    first_available_dim: Optional[int] = None,
) -> Union[EnumMessenger, Callable[_P, _T]]: ...


@overload
def escape(
    escape_fn: Callable[["Message"], bool],
) -> EscapeMessenger: ...


@overload
def escape(
    fn: Callable[_P, _T],
    escape_fn: Callable[["Message"], bool],
) -> Callable[_P, _T]: ...


@_make_handler(EscapeMessenger)
def escape(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    escape_fn: Callable[["Message"], bool],
) -> Union[EscapeMessenger, Callable[_P, _T]]: ...


@overload
def equalize(
    sites: Union[str, List[str]],
    type: Optional[str],
    keep_dist: Optional[bool],
) -> EqualizeMessenger: ...


@overload
def equalize(
    fn: Callable[_P, _T],
    sites: Union[str, List[str]],
    type: Optional[str],
    keep_dist: Optional[bool],
) -> Callable[_P, _T]: ...


@_make_handler(EqualizeMessenger)
def equalize(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    sites: Union[str, List[str]],
    type: Optional[str],
    keep_dist: Optional[bool],
) -> Union[EqualizeMessenger, Callable[_P, _T]]: ...


@overload
def infer_config(
    config_fn: Callable[["Message"], "InferDict"],
) -> InferConfigMessenger: ...


@overload
def infer_config(
    fn: Callable[_P, _T],
    config_fn: Callable[["Message"], "InferDict"],
) -> Callable[_P, _T]: ...


@_make_handler(InferConfigMessenger)
def infer_config(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    config_fn: Callable[["Message"], "InferDict"],
) -> Union[InferConfigMessenger, Callable[_P, _T]]: ...


@overload
def lift(
    prior: Union[Callable, "Distribution", Dict[str, Union["Distribution", Callable]]],
) -> LiftMessenger: ...


@overload
def lift(
    fn: Callable[_P, _T],
    prior: Union[Callable, "Distribution", Dict[str, Union["Distribution", Callable]]],
) -> Callable[_P, _T]: ...


@_make_handler(LiftMessenger)
def lift(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    prior: Union[Callable, "Distribution", Dict[str, Union["Distribution", Callable]]],
) -> Union[LiftMessenger, Callable[_P, _T]]: ...


@overload
def mask(
    mask: Union[bool, "torch.BoolTensor"],
) -> MaskMessenger: ...


@overload
def mask(
    fn: Callable[_P, _T],
    mask: Union[bool, "torch.BoolTensor"],
) -> Callable[_P, _T]: ...


@_make_handler(MaskMessenger)
def mask(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    mask: Union[bool, "torch.BoolTensor"],
) -> Union[MaskMessenger, Callable[_P, _T]]: ...


@overload
def reparam(
    config: Union[Dict[str, "Reparam"], Callable[["Message"], Optional["Reparam"]]],
) -> ReparamMessenger: ...


@overload
def reparam(
    fn: Callable[_P, _T],
    config: Union[Dict[str, "Reparam"], Callable[["Message"], Optional["Reparam"]]],
) -> ReparamHandler[_P, _T]: ...


@_make_handler(ReparamMessenger)
def reparam(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    config: Union[Dict[str, "Reparam"], Callable[["Message"], Optional["Reparam"]]],
) -> Union[ReparamMessenger, ReparamHandler[_P, _T]]: ...


@overload
def replay(
    fn: None = ...,
    trace: Optional["Trace"] = None,
    params: Optional[Dict[str, "torch.Tensor"]] = None,
) -> ReplayMessenger: ...


@overload
def replay(
    fn: Callable[_P, _T],
    trace: Optional["Trace"] = None,
    params: Optional[Dict[str, "torch.Tensor"]] = None,
) -> Callable[_P, _T]: ...


@_make_handler(ReplayMessenger)
def replay(  # type: ignore[empty-body]
    fn: Optional[Callable[_P, _T]] = None,
    trace: Optional["Trace"] = None,
    params: Optional[Dict[str, "torch.Tensor"]] = None,
) -> Union[ReplayMessenger, Callable[_P, _T]]: ...


@overload
def scale(
    scale: Union[float, "torch.Tensor"],
) -> ScaleMessenger: ...


@overload
def scale(
    fn: Callable[_P, _T],
    scale: Union[float, "torch.Tensor"],
) -> Callable[_P, _T]: ...


@_make_handler(ScaleMessenger)
def scale(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    scale: Union[float, "torch.Tensor"],
) -> Union[ScaleMessenger, Callable[_P, _T]]: ...


@overload
def seed(
    rng_seed: int,
) -> SeedMessenger: ...


@overload
def seed(
    fn: Callable[_P, _T],
    rng_seed: int,
) -> Callable[_P, _T]: ...


@_make_handler(SeedMessenger)
def seed(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    rng_seed: int,
) -> Union[SeedMessenger, Callable[_P, _T]]: ...


@overload
def substitute(
    data: Dict[str, "torch.Tensor"],
) -> SubstituteMessenger: ...


@overload
def substitute(
    fn: Callable[_P, _T],
    data: Dict[str, "torch.Tensor"],
) -> Callable[_P, _T]: ...


@_make_handler(SubstituteMessenger)
def substitute(  # type: ignore[empty-body]
    fn: Callable[_P, _T],
    data: Dict[str, "torch.Tensor"],
) -> Union[SubstituteMessenger, Callable[_P, _T]]: ...


@overload
def trace(
    fn: None = ...,
    graph_type: Optional[Literal["flat", "dense"]] = None,
    param_only: Optional[bool] = None,
) -> TraceMessenger: ...


@overload
def trace(
    fn: Callable[_P, _T],
    graph_type: Optional[Literal["flat", "dense"]] = None,
    param_only: Optional[bool] = None,
) -> TraceHandler[_P, _T]: ...


@_make_handler(TraceMessenger)
def trace(  # type: ignore[empty-body]
    fn: Optional[Callable[_P, _T]] = None,
    graph_type: Optional[Literal["flat", "dense"]] = None,
    param_only: Optional[bool] = None,
) -> Union[TraceMessenger, TraceHandler[_P, _T]]: ...


@overload
def uncondition(
    fn: None = ...,
) -> UnconditionMessenger: ...


@overload
def uncondition(
    fn: Callable[_P, _T] = ...,
) -> Callable[_P, _T]: ...


@_make_handler(UnconditionMessenger)
def uncondition(  # type: ignore[empty-body]
    fn: Optional[Callable[_P, _T]] = None,
) -> Union[UnconditionMessenger, Callable[_P, _T]]: ...


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
    fn: Optional[Union[Iterable[int], Callable[_P, _T]]] = None,
    history: int = 1,
    keep: bool = False,
    dim: Optional[int] = None,
    name: Optional[str] = None,
) -> Union[MarkovMessenger, Callable[_P, _T]]:
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
