from pyro.params.param_store import ParamStoreDict
from torch.autograd import Variable
from pyro.optim.optim import PyroOptim
from inspect import isclass
import pyro
from torch.nn import Parameter
import torch

from pyro import distributions, infer, nn, params, util, poutine

from pyro.util import zeros, ones
from pyro.params import param_with_module_name

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


##############################################################
# New primitive definitions with bidirectional stack traversal
##############################################################
def apply_stack(initial_msg, stack=None):
    """
    execute the poutine stack according to the new two-sided blocking scheme
    New Poutine stack mechanism:
    1) start at the top
    2) grab the top poutine, ask to go down
    3) if down, recur
    4) if not, stop, start returning
    """
    if stack is None:
        # XXX what should be referenced here?
        stack = _PYRO_STACK

    # # XXX seems like this should happen on poutine installation, not at execution
    # assert poutine.validate_stack(stack), \
    #     "Current poutine stack violates poutine composition rules"

    msg = initial_msg

    # work out the bottom poutine at this site
    for i in range(len(stack) - 1, -1, -1):
        msg, stop = stack[i].down(msg)
        if stop:
            break

    # go until time to stop?
    for j in range(i, len(stack)):
        msg, stop = stack[j].up(msg)
        if stop:
            break

    return msg


def sample(name, fn, *args, **kwargs):
    """
    current sample interface
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
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["ret"]


def observe(name, fn, val, *args, **kwargs):
    """
    current observe interface
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
            "val": val,
            "args": args,
            "kwargs": kwargs,
            "ret": None,
            "scale": 1.0,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["ret"]


def map_data(name, data, fn, batch_size=None):
    """
    current map_data interface
    """
    if len(_PYRO_STACK) == 0:
        # default behavior
        if isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
            if batch_size > 0:
                scale = float(data.size(0)) / float(batch_size)
                ind = Variable(torch.randperm(data.size(0))[0:batch_size])
                ind_data = data.index_select(0, ind)
            else:
                # if batch_size == 0, don't index (saves time/space)
                scale = 1.0
                ind = Variable(torch.arange(0, data.size(0)))
                ind_data = data
        else:
            # if batch_size > 0, select a random set of indices and store it
            if batch_size > 0:
                ind = torch.randperm(len(data))[0:batch_size].numpy().tolist()
                scale = float(len(data)) / float(batch_size)
                ind_data = [data[i] for i in ind]
            else:
                ind = list(range(len(data)))
                scale = 1.0
                ind_data = data

        if isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
            ret = fn(ind, ind_data)
        else:
            ret = list(map(lambda ix: fn(*ix), enumerate(ind_data)))
        return ret
    else:
        # initialize data structure to pass up/down the stack
        msg = {
            "type": "map_data",
            "name": name,
            "fn": fn,
            "data": data,
            "batch_size": batch_size,
            # XXX should these be added here or during application
            "indices": None,
            "scale": None,
            "ret": None,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["ret"]


# XXX this should have the same call signature as torch.Tensor constructors
def param(name, *args, **kwargs):
    """
    New version of param based on updated poutine stack logic
    """
    if len(_PYRO_STACK) == 0:
        return _param_store.get_param(name, *args, **kwargs)
    else:
        msg = {
            "type": "param",
            "name": name,
            "args": args,
            "kwargs": kwargs,
            "ret": None,
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
