from pyro.params.param_store import _MODULE_NAMESPACE_DIVIDER, ParamStoreDict  # noqa: F401

# the global pyro stack
_PYRO_STACK = []

# the global ParamStore
_PYRO_PARAM_STORE = ParamStoreDict()


class _DimAllocator(object):
    """
    Dimension allocator for internal use by :class:`iarange`.
    There is a single global instance.

    Note that dimensions are indexed from the right, e.g. -1, -2.
    """
    def __init__(self):
        self._stack = []  # in reverse orientation of log_prob.shape

    def allocate(self, name, dim):
        """
        Allocate a dimension to an :class:`iarange` with given name.
        Dim should be either None for automatic allocation or a negative
        integer for manual allocation.
        """
        if name in self._stack:
            raise ValueError('duplicate iarange "{}"'.format(name))
        if dim is None:
            # Automatically allocate the rightmost dimension to the left of all existing dims.
            self._stack.append(name)
            dim = -len(self._stack)
        elif dim >= 0:
            raise ValueError('Expected dim < 0 to index from the right, actual {}'.format(dim))
        else:
            # Allocate the requested dimension.
            while dim < -len(self._stack):
                self._stack.append(None)
            if self._stack[-1 - dim] is not None:
                raise ValueError('\n'.join([
                    'at iaranges "{}" and "{}", collide at dim={}'.format(name, self._stack[-1 - dim], dim),
                    '\nTry moving the dim of one iarange to the left, e.g. dim={}'.format(dim - 1)]))
            self._stack[-1 - dim] = name
        return dim

    def free(self, name, dim):
        """
        Free a dimension.
        """
        assert self._stack[-1 - dim] == name
        self._stack[-1 - dim] = None
        while self._stack and self._stack[-1] is None:
            self._stack.pop()


# Handles placement of enumeration and independence dimensions
_DIM_ALLOCATOR = _DimAllocator()


class NonlocalExit(Exception):
    """
    Exception for exiting nonlocally from poutine execution.

    Used by poutine.EscapeMessenger to return site information.
    """
    def __init__(self, site, *args, **kwargs):
        """
        :param site: message at a pyro site

        constructor.  Just stores the input site.
        """
        super(NonlocalExit, self).__init__(*args, **kwargs)
        self.site = site

    def reset_stack(self):
        """
        Reset the state of the frames remaining in the stack.
        Necessary for multiple re-executions in poutine.queue.
        """
        for frame in _PYRO_STACK:
            frame._reset()


def validate_message(msg):
    """
    Asserts that the message has a valid format.
    :returns: None
    """
    assert msg["type"] in ("sample", "param"), \
        "{} is an invalid site type, how did that get there?".format(msg["type"])


def default_process_message(msg):
    """
    Default method for processing messages in inference.
    :param msg: a message to be processed
    :returns: None
    """
    validate_message(msg)
    if msg["type"] == "sample":
        fn, args, kwargs = \
            msg["fn"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg

        if msg["is_observed"]:
            assert msg["value"] is not None
            val = msg["value"]
        else:
            val = fn(*args, **kwargs)

        # after fn has been called, update msg to prevent it from being called again.
        msg["done"] = True
        msg["value"] = val
    elif msg["type"] == "param":
        name, args, kwargs = \
            msg["name"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg

        ret = _PYRO_PARAM_STORE.get_param(name, *args, **kwargs)

        # after the param store has been queried, update msg["done"]
        # to prevent it from being queried again.
        msg["done"] = True
        msg["value"] = ret
    else:
        assert False
    return None


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

    counter = 0
    # go until time to stop?
    for frame in stack:
        validate_message(msg)

        counter = counter + 1

        frame._process_message(msg)

        if msg["stop"]:
            break

    default_process_message(msg)

    for frame in reversed(stack[0:counter]):
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
