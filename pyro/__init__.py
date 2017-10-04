from inspect import isclass

import torch
from torch.autograd import Variable
from torch.nn import Parameter

import pyro
from pyro import util
from pyro.optim.optim import PyroOptim
from pyro.params import param_with_module_name
from pyro.params.param_store import ParamStoreDict
from pyro.util import zeros, ones, set_rng_seed  # noqa: F401

# global map of params for now
_param_store = ParamStoreDict()

# used to create fully-formed param names, e.g. mymodule$$$mysubmodule.weight
_MODULE_NAMESPACE_DIVIDER = "$$$"


def get_param_store():
    """
    Returns the param store
    """

    return _param_store


def device(x):
    """
    :param x: Pytorch tensor or Variable
    :type: Pytorch Tensor
    :returns: Pytorch tensor or Variable

    Returns CUDATensor is CUDA is enabled
    """
    if torch.cuda.is_available():
        return x.cuda()
    return x.cpu()


# use pyro optim class to wrap nn optim
optim = PyroOptim

_PYRO_STACK = []


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

    # first, gather all information necessary to apply the stack to this site
    for frame in reversed(stack):
        msg = frame.down(msg)

    # go until time to stop?
    for frame in stack:
        assert msg["type"] in ("sample", "observe", "map_data", "param"), \
            "{} is an invalid site type, how did that get there?".format(msg["type"])

        msg.update({"ret": getattr(frame, "_pyro_{}".format(msg["type"]))(msg)})

        if msg["stop"]:
            break

    return msg


def sample(name, fn, *args, **kwargs):
    """
    :param name: name of sample
    :param fn: distribution class or function
    :returns: sample

    Samples from the distribution and registers it in the trace data structure.
    """
    # check if stack is empty
    # if stack empty, default behavior (defined here)
    if len(_PYRO_STACK) == 0:
        return fn(*args, **kwargs)
    # if stack not empty, apply everything in the stack?
    else:
        # initialize data structure to pass up/down the stack
        msg = {
            "type": "sample",
            "name": name,
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "ret": None,
            "scale": 1.0,
            "map_data_stack": [],
            # "done": False,
            "stop": False,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["ret"]


def observe(name, fn, obs, *args, **kwargs):
    """
    :param name: name of observation
    :param fn: distribution class or function
    :param obs: observed datum
    :returns: sample

    Only should be used in the context of inference.
    Calculates the score of the sample and registers
    it in the trace data structure.
    """
    if len(_PYRO_STACK) == 0:
        raise NotImplementedError(
            "Observe has been used outside of a normalizing context.")
    else:
        # initialize data structure to pass up/down the stack
        msg = {
            "type": "observe",
            "name": name,
            "fn": fn,
            "obs": obs,
            "args": args,
            "kwargs": kwargs,
            "ret": None,
            "scale": 1.0,
            "map_data_stack": [],
            # "done": False,
            "stop": False,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["ret"]


def map_data(name, data, fn, batch_size=0, batch_dim=0):
    """
    Data subsampling with the important property that all the data are conditionally independent.

    With default values of `batch_size` and `batch_dim`, `map_data` behaves like `map`.
    More precisely, `map_data('foo', data, fn)` is equivalent to `[fn(i, x) for i, x in enumerate(data)]`.

    :param str name: named argument
    :param data: data to subsample
    :param callable fn: a function taking `(index, datum)` pairs, where `dataum = data[index]`
    :param int batch_size: number of samples per batch, or zero for the entire dataset
    :param int batch_dim: dimension to subsample for tensor inputs
    :return: a list of values returned by `fn`
    """
    if len(_PYRO_STACK) == 0:
        # default behavior
        ind = util.get_batch_indices(data, batch_size, batch_dim)
        if batch_size == 0:
            ind_data = data
        elif isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
            ind_data = data.index_select(batch_dim, ind)
        else:
            ind_data = [data[i] for i in ind]

        if isinstance(data, (torch.Tensor, Variable)):
            ret = fn(ind, ind_data)
        else:
            ret = [fn(i, x) for i, x in zip(ind, ind_data)]
        return ret
    else:
        # initialize data structure to pass up/down the stack
        msg = {
            "type": "map_data",
            "name": name,
            "fn": fn,
            "data": data,
            "batch_size": batch_size,
            "batch_dim": batch_dim,
            # XXX should these be added here or during application
            "indices": None,
            "ret": None,
            "done": False,
            "stop": False,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["ret"]


# XXX this should have the same call signature as torch.Tensor constructors
def param(name, *args, **kwargs):
    """
    :param name: name of parameter
    :returns: parameter

    Saves the variable as a parameter in the param store.
    To interact with the param store or write to disk,
    see `Parameters <parameters.html>`_.
    """
    if len(_PYRO_STACK) == 0:
        return _param_store.get_param(name, *args, **kwargs)
    else:
        msg = {
            "type": "param",
            "name": name,
            "args": args,
            "kwargs": kwargs,
            "scale": 1.0,
            "ret": None,
            "stop": False,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["ret"]


# hand off behavior to poutine if necessary?
# for now default calls out to pyro.param -- which is handled by poutine
def module(pyro_name, nn_obj):
    """
    :param pyro_name: name of module
    :param nn_obj: pytorch nn module
    :returns: pytorch nn object

    Takes a pytorch nn module and registers its parameters with the param store.
    In conjunction with the param store save() and load() functionality, this
    allows the user to save and load nn modules
    """
    assert hasattr(nn_obj, "parameters"), "module has no parameters"
    assert _MODULE_NAMESPACE_DIVIDER not in pyro_name, "improper module name, since contains %s" %\
        _MODULE_NAMESPACE_DIVIDER

    if isclass(nn_obj):
        raise NotImplementedError("Not yet supporting class constructor")

    state_dict = {}
    for param_name, param in nn_obj.named_parameters():
        state_dict[param_name] = pyro.param(param_with_module_name(pyro_name, param_name), param)

    current_nn_state = nn_obj.state_dict()
    for name, param in state_dict.items():
        if name not in current_nn_state:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        # only copy if the param has actually changed
        # Note: apart from the following line, the rest of this code
        # logic is borrowed from torch.nn.Module.load_state_dict
        if id(param) != id(current_nn_state[name]):
            current_nn_state[name].copy_(param)

    missing = set(current_nn_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    return nn_obj
