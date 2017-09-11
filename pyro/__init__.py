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
    return _param_store


def device(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x.cpu()


# use pyro optim class to wrap nn optim
optim = PyroOptim

_PYRO_STACK = []


def param(name, *args, **kwargs):
    if len(_PYRO_STACK) == 0:
        return _param_store.get_param(name, *args, **kwargs)
    else:
        ret = None
        for layer in _PYRO_STACK:
            ret, stop = layer("param", ret, name, *args, **kwargs)
            if stop:
                break
        return ret


def sample(name, fn, *args, **kwargs):
    # check if stack is empty
    # if stack empty, default behavior (defined here)
    if len(_PYRO_STACK) == 0:
        return fn(*args, **kwargs)
    # if stack not empty, apply everything in the stack?
    else:
        ret = None
        for layer in _PYRO_STACK:
            ret, stop = layer("sample", ret, name, fn, *args, **kwargs)
            if stop:
                break
        return ret


def observe(name, fn, obs, *args, **kwargs):
    if len(_PYRO_STACK) == 0:
        raise NotImplementedError(
            "Observe has been used outside of a normalizing context.")
    else:
        ret = None
        for layer in _PYRO_STACK:
            ret, stop = layer("observe", ret, name, fn, obs, *args, **kwargs)
            if stop:
                break
        return ret


def map_data(name, data, observer, *args, **kwargs):
    # by default map_data is the same as map.
    # infer algs (eg VI) that do minibatches should overide this.
    if len(_PYRO_STACK) == 0:
        return [observer(i, datum) for i, datum in enumerate(data)]
    else:
        ret = None
        for layer in _PYRO_STACK:
            ret, stop = layer("map_data", ret, name, data, observer, *args, **kwargs)
            if stop:
                break
        return ret

# hand off behavior to poutine if necessary?
# for now default calls out to pyro.param -- which is handled by poutine


def module(pyro_name, nn_obj):
    """
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

def random_module(module, prior, *args, **kwargs):
    assert hasattr(module, "parameters"), "Module is not a NN module."
    return poutine.lift(module, prior, *args, **kwargs)
