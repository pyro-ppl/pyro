from pyro.params.param_store import ParamStoreDict
from torch.autograd import Variable
from pyro.optim.optim import PyroOptim
from inspect import isclass
import pyro
from torch.nn import Parameter
import torch

from pyro.util import zeros, ones

# global map of params for now
_param_store = ParamStoreDict()

# set global tensor type (cpu v.gpu); cpu by default
_global_tensor_type = 'cpu'


def get_param_store():
    return _param_store


def set_cuda():
    global _global_tensor_type
    _global_tensor_type = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def set_cpu():
    global _global_tensor_type
    _global_tensor_type = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')


def device(x):
    if _global_tensor_type == 'cpu':
        return x.cpu()
    elif _global_tensor_type == 'cuda':
        return x.cuda()


module_namespace_divider = "$$$"


def module_name(pyro_name, param_name):
    return module_namespace_divider.join([pyro_name, param_name])


def module_from_param_name(param_name):
    # if param_name is not None:
    return param_name.split(module_namespace_divider)[0]


def user_param_name(param_name):
    if module_namespace_divider in param_name:
        return param_name.split(module_namespace_divider)[1]
    else:
        return param_name


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


def sync_module(pyro_name, nn_obj):
    assert hasattr(nn_obj, "parameters"), "module has no parameters"
    assert module_namespace_divider not in pyro_name, "improper module name, since contains %s" %\
                                                      module_namespace_divider

    state_dict = {}
    for param_name, param in _param_store._params.items():
        if module_namespace_divider in param_name:
            module_name = module_from_param_name(param_name)
            param_key = user_param_name(param_name)
            state_dict[param_key] = param

    nn_obj.load_state_dict(state_dict)


def module(pyro_name, nn_obj):  # :, *args, **kwargs):
    assert hasattr(nn_obj, "parameters"), "module has no parameters"

    # cannot contain our special modifier marker
    assert module_namespace_divider not in pyro_name, "improper module name, since contains %s" %\
                                                      module_namespace_divider

    if isclass(nn_obj):
        raise NotImplementedError("Not yet supporting class constructor")

    # for now, we simply loop through parameters every time and marke then
    for param_name, param in nn_obj.named_parameters():

        # is param a variable? Save variable inside param store
        # XXX NOTE: this won't save persistent buffers
        if isinstance(param, Variable):

            # mark the object
            pyro.param(module_name(pyro_name, param_name), param)

    # send back object for calling forward
    # if we want to intercept somewhere else we can
    return nn_obj
