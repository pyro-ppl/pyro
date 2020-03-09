# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools

from pyro.params.param_store import _MODULE_NAMESPACE_DIVIDER, ParamStoreDict  # noqa: F401

# the global pyro stack
_PYRO_STACK = []

# the global ParamStore
_PYRO_PARAM_STORE = ParamStoreDict()


class _DimAllocator:
    """
    Dimension allocator for internal use by :class:`plate`.
    There is a single global instance.

    Note that dimensions are indexed from the right, e.g. -1, -2.
    """
    def __init__(self):
        self._stack = []  # in reverse orientation of log_prob.shape

    def allocate(self, name, dim):
        """
        Allocate a dimension to an :class:`plate` with given name.
        Dim should be either None for automatic allocation or a negative
        integer for manual allocation.
        """
        if name in self._stack:
            raise ValueError('duplicate plate "{}"'.format(name))
        if dim is None:
            # Automatically designate the rightmost available dim for allocation.
            dim = -1
            while -dim <= len(self._stack) and self._stack[-1 - dim] is not None:
                dim -= 1
        elif dim >= 0:
            raise ValueError('Expected dim < 0 to index from the right, actual {}'.format(dim))

        # Allocate the requested dimension.
        while dim < -len(self._stack):
            self._stack.append(None)
        if self._stack[-1 - dim] is not None:
            raise ValueError('\n'.join([
                'at plates "{}" and "{}", collide at dim={}'.format(name, self._stack[-1 - dim], dim),
                '\nTry moving the dim of one plate to the left, e.g. dim={}'.format(dim - 1)]))
        self._stack[-1 - dim] = name
        return dim

    def free(self, name, dim):
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
    def set_first_available_dim(self, first_available_dim):
        """
        Set the first available dim, which should be to the left of all
        :class:`plate` dimensions, e.g. ``-1 - max_plate_nesting``. This should
        be called once per program. In SVI this should be called only once per
        (guide,model) pair.
        """
        assert first_available_dim < 0, first_available_dim
        self.next_available_dim = first_available_dim
        self.next_available_id = 0
        self.dim_to_id = {}  # only the global ids

    def allocate(self, scope_dims=None):
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
        if dim == -float('inf'):
            raise ValueError("max_plate_nesting must be set to a finite value for parallel enumeration")
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
    def __init__(self, site, *args, **kwargs):
        """
        :param site: message at a pyro site constructor.
            Just stores the input site.
        """
        super().__init__(*args, **kwargs)
        self.site = site

    def reset_stack(self):
        """
        Reset the state of the frames remaining in the stack.
        Necessary for multiple re-executions in poutine.queue.
        """
        for frame in reversed(_PYRO_STACK):
            frame._reset()
            if type(frame).__name__ == "BlockMessenger" and frame.hide_fn(self.site):
                break


def default_process_message(msg):
    """
    Default method for processing messages in inference.

    :param msg: a message to be processed
    :returns: None
    """
    if msg["done"] or msg["is_observed"] or msg["value"] is not None:
        msg["done"] = True
        return msg

    msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    # after fn has been called, update msg to prevent it from being called again.
    msg["done"] = True


def apply_stack(initial_msg):
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

    return None


def am_i_wrapped():
    """
    Checks whether the current computation is wrapped in a poutine.
    :returns: bool
    """
    return len(_PYRO_STACK) > 0


def effectful(fn=None, type=None):
    """
    :param fn: function or callable that performs an effectful computation
    :param str type: the type label of the operation, e.g. `"sample"`

    Wrapper for calling :func:`~pyro.poutine.runtime.apply_stack` to apply any active effects.
    """
    if fn is None:
        return functools.partial(effectful, type=type)

    if getattr(fn, "_is_effectful", None):
        return fn

    assert type is not None, "must provide a type label for operation {}".format(fn)
    assert type != "message", "cannot use 'message' as keyword"

    @functools.wraps(fn)
    def _fn(*args, **kwargs):

        name = kwargs.pop("name", None)
        infer = kwargs.pop("infer", {})

        value = kwargs.pop("obs", None)
        is_observed = value is not None

        if not am_i_wrapped():
            return fn(*args, **kwargs)
        else:
            msg = {
                "type": type,
                "name": name,
                "fn": fn,
                "is_observed": is_observed,
                "args": args,
                "kwargs": kwargs,
                "value": value,
                "scale": 1.0,
                "mask": None,
                "cond_indep_stack": (),
                "done": False,
                "stop": False,
                "continuation": None,
                "infer": infer,
            }
            # apply the stack and return its return value
            apply_stack(msg)
            return msg["value"]
    _fn._is_effectful = True
    return _fn
