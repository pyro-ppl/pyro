from pyro.params.param_store import ParamStoreDict
from .indep_poutine import _DimAllocator


# the global pyro stack
_PYRO_STACK = []

# the global ParamStore
_PYRO_PARAM_STORE = ParamStoreDict()

# used to create fully-formed param names, e.g. mymodule$$$mysubmodule.weight
_MODULE_NAMESPACE_DIVIDER = "$$$"

# Handles placement of enumeration and independence dimensions
_DIM_ALLOCATOR = _DimAllocator()


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
    :param dict initial_msg: the starting version of the trace site
    :returns: an updated message that is the final version of the trace site

    Execute the poutine stack at a single site according to the following scheme:
    1. Walk down the stack from top to bottom, collecting into the message
        all information necessary to execute the stack at that site
    2. For each poutine in the stack from bottom to top:
           Execute the poutine with the message;
           If the message field "stop" is True, stop;
           Otherwise, continue
    3. Return the updated message
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
