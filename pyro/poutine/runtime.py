# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from typing_extensions import Literal, ParamSpec, TypedDict

from pyro.params.param_store import (  # noqa: F401
    _MODULE_NAMESPACE_DIVIDER,
    ParamStoreDict,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")

if TYPE_CHECKING:
    from collections import Counter

    from pyro.distributions.score_parts import ScoreParts
    from pyro.distributions.torch_distribution import TorchDistributionMixin
    from pyro.poutine.indep_messenger import CondIndepStackFrame
    from pyro.poutine.messenger import Messenger

# the global pyro stack
_PYRO_STACK: List["Messenger"] = []

# the global ParamStore
_PYRO_PARAM_STORE = ParamStoreDict()


class InferDict(TypedDict, total=False):
    """
    A dictionary that contains information about inference.

    This can be used to configure per-site inference strategies, e.g.::

        pyro.sample(
            "x",
            dist.Bernoulli(0.5),
            infer={"enumerate": "parallel"},
        )

    Keys:
        enumerate (str):
            If one of the strings "sequential" or "parallel", enables
            enumeration. Parallel enumeration is generally faster but requires
            broadcasting-safe operations and static structure.
        expand (bool):
            Whether to expand the distribution during enumeration. Defaults to
            False if missing.
        is_auxiliary (bool):
            Whether the sample site is auxiliary, e.g. for use in guides that
            deterministically transform auxiliary variables. Defaults to False
            if missing.
        is_observed (bool):
            Whether the sample site is observed (i.e. not latent). Defaults to
            False if missing.
        num_samples (int):
            The number of samples to draw. Defaults to 1 if missing.
        obs (optional torch.Tensor):
            The observed value, or None for latent variables. Defaults to None
            if missing.
        prior (optional torch.distributions.Distribution):
            (internal) For use in GuideMessenger to store the model's prior
            distribution (conditioned on upstream sites).
        tmc (str):
            Whether to use the diagonal or mixture approximation for Tensor
            Monte Carlo in TraceTMC_ELBO.
        was_observed (bool):
            (internal) Whether the sample site was originally observed, in the
            context of inference via Reweighted Wake Sleep or Compiled
            Sequential Importance Sampling.
    """

    enumerate: Literal["sequential", "parallel"]
    expand: bool
    is_auxiliary: bool
    is_observed: bool
    num_samples: int
    obs: Optional[torch.Tensor]
    prior: "TorchDistributionMixin"
    tmc: Literal["diagonal", "mixture"]
    was_observed: bool
    _deterministic: bool
    _dim_to_symbol: Dict[int, str]
    _do_not_trace: bool
    _enumerate_symbol: str
    _markov_scope: "Counter"
    _enumerate_dim: int
    _dim_to_id: Dict[int, int]
    _markov_depth: int


class Message(TypedDict, Generic[_P, _T], total=False):
    """
    Pyro's internal message type for effect handling.

    Messages are stored in trace objects, e.g.::

        trace.nodes["my_site_name"]  # This is a Message.

    Keys:
        type (str):
            The message type, typically one of the strings "sample", "param",
            "plate", or "markov", but possibly custom.
        name (str):
            The site name, typically naming a sample or parameter.
        fn (callable):
            The distribution or function used to generate the sample.
        is_observed (bool):
            A flag to indicate whether the value is observed.
        args (tuple):
            Positional arguments to the distribution or function.
        kwargs (dict):
            Keyword arguments to the distribution or function.
        value (torch.Tensor):
            The value of the sample (either observed or sampled).
        scale (torch.Tensor):
            A scaling factor for the log probability.
        mask (bool torch.Tensor):
            A bool or tensor to mask the log probability.
        cond_indep_stack (tuple):
            The site's local stack of conditional independence metadata.
            Immutable.
        done (bool):
            A flag to indicate whether the message has been handled.
        stop (bool):
            A flag to stop further processing of the message.
        continuation (callable):
            A function to call after processing the message.
        infer (optional InferDict):
            A dictionary of inference parameters.
        obs (torch.Tensor):
            The observed value.
        log_prob (torch.Tensor):
            The log probability of the sample.
        log_prob_sum (torch.Tensor):
            The sum of the log probability.
        unscaled_log_prob (torch.Tensor):
            The unscaled log probability.
        score_parts (pyro.distributions.ScoreParts):
            A collection of score parts.
        packed (Message):
            A packed message, used during enumeration.
    """

    type: str
    name: Optional[str]
    fn: Callable[_P, _T]
    is_observed: bool
    args: Tuple
    kwargs: Dict
    value: Optional[_T]
    scale: Union[torch.Tensor, float]
    mask: Union[bool, torch.Tensor, None]
    cond_indep_stack: Tuple["CondIndepStackFrame", ...]
    done: bool
    stop: bool
    continuation: Optional[Callable[["Message"], None]]
    infer: Optional[InferDict]
    obs: Optional[torch.Tensor]
    log_prob: torch.Tensor
    log_prob_sum: torch.Tensor
    unscaled_log_prob: torch.Tensor
    score_parts: "ScoreParts"
    packed: "Message"
    _intervener_id: Optional[str]


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


@overload
def effectful(
    fn: None = ..., type: Optional[str] = ...
) -> Callable[[Callable[_P, _T]], Callable[..., _T]]: ...


@overload
def effectful(
    fn: Callable[_P, _T] = ..., type: Optional[str] = ...
) -> Callable[..., _T]: ...


def effectful(
    fn: Optional[Callable[_P, _T]] = None, type: Optional[str] = None
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
        *args: _P.args,
        name: Optional[str] = None,
        infer: Optional[InferDict] = None,
        obs: Optional[_T] = None,
        **kwargs: _P.kwargs,
    ) -> _T:
        is_observed = obs is not None

        if not am_i_wrapped():
            return fn(*args, **kwargs)
        else:
            msg = Message(
                type=type,
                name=name,
                fn=fn,
                is_observed=is_observed,
                args=args,
                kwargs=kwargs,
                value=obs,
                scale=1.0,
                mask=None,
                cond_indep_stack=(),
                done=False,
                stop=False,
                continuation=None,
                infer=infer if infer is not None else {},
            )
            # apply the stack and return its return value
            apply_stack(msg)
            if TYPE_CHECKING:
                assert msg["value"] is not None
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
    msg = Message(
        type="inspect",
        name="_pyro_inspect",
        fn=lambda: True,
        is_observed=False,
        args=(),
        kwargs={},
        value=None,
        infer={"_do_not_trace": True},
        scale=1.0,
        mask=None,
        cond_indep_stack=(),
        done=False,
        stop=False,
        continuation=None,
    )
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


def get_plates() -> Tuple["CondIndepStackFrame", ...]:
    """
    Records the effects of enclosing ``pyro.plate`` contexts.

    :returns: A tuple of
        :class:`pyro.poutine.indep_messenger.CondIndepStackFrame` objects.
    :rtype: tuple
    """
    return _inspect()["cond_indep_stack"]
