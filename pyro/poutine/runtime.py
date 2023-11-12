# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# overload,
import torch
from typing_extensions import ParamSpec, TypedDict

from pyro.params.param_store import (  # noqa: F401
    _MODULE_NAMESPACE_DIVIDER,
    ParamStoreDict,
)

P = ParamSpec("P")
T = TypeVar("T")

if TYPE_CHECKING:
    from pyro.poutine.indep_messenger import CondIndepStackFrame
    from pyro.poutine.messenger import Messenger

# the global pyro stack
_PYRO_STACK: List[Messenger] = []

# the global ParamStore
_PYRO_PARAM_STORE = ParamStoreDict()


class Message(TypedDict, total=False):
    type: str
    name: Optional[str]
    fn: Callable
    is_observed: bool
    args: Tuple
    kwargs: Dict
    value: Optional[torch.Tensor]
    scale: float
    mask: Union[bool, torch.Tensor, None]
    cond_indep_stack: Tuple[CondIndepStackFrame, ...]
    done: bool
    stop: bool
    continuation: Optional[Callable[[Message], None]]
    infer: Optional[Dict[str, Union[str, bool]]]
    obs: Optional[torch.Tensor]


class _DimAllocator:
    """
    Dimension allocator for internal use by :class:`plate`.
    There is a single global instance.

    Note that dimensions are indexed from the right, e.g. -1, -2.
    """

    def __init__(self) -> None:
        # in reverse orientation of log_prob.shape
        self._stack: List[Optional[str]] = []

    def allocate(self, name: str, dim: Optional[int]) -> int:
        """
        Allocate a dimension to an :class:`plate` with given name.
        Dim should be either None for automatic allocation or a negative
        integer for manual allocation.
        """
        if name in self._stack:
            raise ValueError(f"duplicate plate '{name}'")
        if dim is None:
            # Automatically designate the rightmost available dim for allocation.
            dim = -1
            while -dim <= len(self._stack) and self._stack[-1 - dim] is not None:
                dim -= 1
        elif dim >= 0:
            raise ValueError(f"Expected dim < 0 to index from the right, actual {dim}")

        # Allocate the requested dimension.
        while dim < -len(self._stack):
            self._stack.append(None)
        if self._stack[-1 - dim] is not None:
            raise ValueError(
                "\n".join(
                    [
                        'at plates "{}" and "{}", collide at dim={}'.format(
                            name, self._stack[-1 - dim], dim
                        ),
                        "\nTry moving the dim of one plate to the left, e.g. dim={}".format(
                            dim - 1
                        ),
                    ]
                )
            )
        self._stack[-1 - dim] = name
        return dim

    def free(self, name: str, dim: int) -> None:
        """
        Free a dimension.
        """
        free_idx = -1 - dim  # stack index to free
        assert self._stack[free_idx] == name
        self._stack[free_idx] = None
        while self._stack and self._stack[-1] is None:
            self._stack.pop()


# Handles placement of plate dimensions
_DIM_ALLOCATOR = _DimAllocator()


class _EnumAllocator:
    """
    Dimension allocator for internal use by :func:`~pyro.poutine.markov`.
    There is a single global instance.

    Note that dimensions are indexed from the right, e.g. -1, -2.
    Note that ids are simply nonnegative integers here.
    """

    def set_first_available_dim(self, first_available_dim: int) -> None:
        """
        Set the first available dim, which should be to the left of all
        :class:`plate` dimensions, e.g. ``-1 - max_plate_nesting``. This should
        be called once per program. In SVI this should be called only once per
        (guide,model) pair.
        """
        assert first_available_dim < 0, first_available_dim
        self.next_available_dim = first_available_dim
        self.next_available_id = 0
        self.dim_to_id: Dict[int, int] = {}  # only the global ids

    def allocate(self, scope_dims: Optional[Set[int]] = None) -> Tuple[int, int]:
        """
        Allocate a new recyclable dim and a unique id.

        If ``scope_dims`` is None, this allocates a global enumeration dim
        that will never be recycled. If ``scope_dims`` is specified, this
        allocates a local enumeration dim that can be reused by at any other
        local site whose scope excludes this site.

        :param set scope_dims: An optional set of (negative integer)
            local enumeration dims to avoid when allocating this dim.
        :return: A pair ``(dim, id)``, where ``dim`` is a negative integer
            and ``id`` is a nonnegative integer.
        :rtype: tuple
        """
        id_ = self.next_available_id
        self.next_available_id += 1

        dim = self.next_available_dim
        if dim == -float("inf"):
            raise ValueError(
                "max_plate_nesting must be set to a finite value for parallel enumeration"
            )
        if scope_dims is None:
            # allocate a new global dimension
            self.next_available_dim -= 1
            self.dim_to_id[dim] = id_
        else:
            # allocate a new local dimension
            while dim in scope_dims:
                dim -= 1

        return dim, id_


# Handles placement of enumeration dimensions
_ENUM_ALLOCATOR = _EnumAllocator()


class NonlocalExit(Exception):
    """
    Exception for exiting nonlocally from poutine execution.

    Used by poutine.EscapeMessenger to return site information.
    """

    def __init__(self, site: Message, *args, **kwargs) -> None:
        """
        :param site: message at a pyro site constructor.
            Just stores the input site.
        """
        super().__init__(*args, **kwargs)
        self.site = site

    def reset_stack(self) -> None:
        """
        Reset the state of the frames remaining in the stack.
        Necessary for multiple re-executions in poutine.queue.
        """
        from pyro.poutine.block_messenger import BlockMessenger

        for frame in reversed(_PYRO_STACK):
            frame._reset()
            if isinstance(frame, BlockMessenger) and frame.hide_fn(self.site):
                break


def default_process_message(msg: Message) -> None:
    """
    Default method for processing messages in inference.

    :param msg: a message to be processed
    :returns: None
    """
    if msg["done"] or msg["is_observed"] or msg["value"] is not None:
        msg["done"] = True
        return

    msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    # after fn has been called, update msg to prevent it from being called again.
    msg["done"] = True


def apply_stack(initial_msg: Message) -> None:
    """
    Execute the effect stack at a single site according to the following scheme:

        1. For each ``Messenger`` in the stack from bottom to top,
           execute ``Messenger._process_message`` with the message;
           if the message field "stop" is True, stop;
           otherwise, continue
        2. Apply default behavior (``default_process_message``) to finish remaining site execution
        3. For each ``Messenger`` in the stack from top to bottom,
           execute ``_postprocess_message`` to update the message and internal messenger state with the site results
        4. If the message field "continuation" is not ``None``, call it with the message

    :param dict initial_msg: the starting version of the trace site
    :returns: ``None``
    """
    stack = _PYRO_STACK
    # TODO check at runtime if stack is valid

    # msg is used to pass information up and down the stack
    msg = initial_msg

    pointer = 0
    # go until time to stop?
    for frame in reversed(stack):
        pointer = pointer + 1

        frame._process_message(msg)

        if msg["stop"]:
            break

    default_process_message(msg)

    for frame in stack[-pointer:]:
        frame._postprocess_message(msg)

    cont = msg["continuation"]
    if cont is not None:
        cont(msg)


def am_i_wrapped() -> bool:
    """
    Checks whether the current computation is wrapped in a poutine.
    :returns: bool
    """
    return len(_PYRO_STACK) > 0


#  @overload
#  def effectful(
#      fn: None = ..., type: Optional[str] = ...
#  ) -> Callable[[Optional[Callable[P, T]]], Callable[P, Union[T, torch.Tensor, None]]]:
#      ...
#
#
#  @overload
#  def effectful(
#      fn: Callable[P, T] = ..., type: Optional[str] = ...
#  ) -> Callable[P, Union[T, torch.Tensor, None]]:
#      ...


def effectful(
    fn: Optional[Callable[P, T]] = None, type: Optional[str] = None
) -> Callable:
    """
    :param fn: function or callable that performs an effectful computation
    :param str type: the type label of the operation, e.g. `"sample"`

    Wrapper for calling :func:`~pyro.poutine.runtime.apply_stack` to apply any active effects.
    """
    if fn is None:
        return functools.partial(effectful, type=type)

    if getattr(fn, "_is_effectful", None):
        return fn

    assert type is not None, f"must provide a type label for operation {fn}"
    assert type != "message", "cannot use 'message' as keyword"

    @functools.wraps(fn)
    def _fn(
        *args: P.args,
        name: Optional[str] = None,
        infer: Optional[Dict] = None,
        obs: Optional[torch.Tensor] = None,
        **kwargs: P.kwargs,
    ) -> Union[T, torch.Tensor, None]:
        is_observed = obs is not None

        if not am_i_wrapped():
            return fn(*args, **kwargs)
        else:
            msg: Message = {
                "type": type,
                "name": name,
                "fn": fn,
                "is_observed": is_observed,
                "args": args,
                "kwargs": kwargs,
                "value": obs,
                "scale": 1.0,
                "mask": None,
                "cond_indep_stack": (),
                "done": False,
                "stop": False,
                "continuation": None,
                "infer": infer if infer is not None else {},
            }
            # apply the stack and return its return value
            apply_stack(msg)
            return msg["value"]

    _fn._is_effectful = True  # type: ignore[attr-defined]
    return _fn


def _inspect() -> Message:
    """
    EXPERIMENTAL Inspect the Pyro stack.

    .. warning:: The format of the returned message may change at any time and
        does not guarantee backwards compatibility.

    :returns: A message with all effects applied.
    :rtype: dict
    """
    msg: Message = {
        "type": "inspect",
        "name": "_pyro_inspect",
        "fn": lambda: True,
        "is_observed": False,
        "args": (),
        "kwargs": {},
        "value": None,
        "infer": {"_do_not_trace": True},
        "scale": 1.0,
        "mask": None,
        "cond_indep_stack": (),
        "done": False,
        "stop": False,
        "continuation": None,
    }
    apply_stack(msg)
    return msg


def get_mask() -> Union[bool, torch.Tensor, None]:
    """
    Records the effects of enclosing ``poutine.mask`` handlers.

    This is useful for avoiding expensive ``pyro.factor()`` computations during
    prediction, when the log density need not be computed, e.g.::

        def model():
            # ...
            if poutine.get_mask() is not False:
                log_density = my_expensive_computation()
                pyro.factor("foo", log_density)
            # ...

    :returns: The mask.
    :rtype: None, bool, or torch.Tensor
    """
    return _inspect()["mask"]


def get_plates() -> Tuple[CondIndepStackFrame, ...]:
    """
    Records the effects of enclosing ``pyro.plate`` contexts.

    :returns: A tuple of
        :class:`pyro.poutine.indep_messenger.CondIndepStackFrame` objects.
    :rtype: tuple
    """
    return _inspect()["cond_indep_stack"]
