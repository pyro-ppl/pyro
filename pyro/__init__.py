from pyro.params.param_store import ParamStoreDict
from torch.autograd import Variable
from pyro.optim.optim import PyroOptim
from inspect import isclass
import pyro
from torch.nn import Parameter
import torch

# global map of params for now
_param_store = ParamStoreDict()

# set pyro.param function to _param_store.get_param
param = _param_store.get_param

# set global tensor type (cpu v.gpu); cpu by default
_global_tensor_type = 'cpu'


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


mod_div = "$$$"


def module_name(pyro_name, param):
    return mod_div.join([pyro_name, str(id(param))])


def module_from_param_name(param_name):
    if param_name is not None:
        return param_name.split(mod_div)[0]


# use pyro optim class to wrap nn optim
optim = PyroOptim


def ones(*args, **kwargs):
    return Parameter(torch.ones(*args, **kwargs))
    # return pyro.device(Parameter(torch.ones(*args, **kwargs)))


def zeros(*args, **kwargs):
    return Parameter(torch.zeros(*args, **kwargs))
    # return pyro.device(Parameter(torch.zeros(*args, **kwargs)))


def ng_ones(*args, **kwargs):
    return Variable(torch.ones(*args, **kwargs), requires_grad=False)


def ng_zeros(*args, **kwargs):
    return Variable(torch.zeros(*args, **kwargs), requires_grad=False)


def sample(name, dist, *args, **kwargs):
    """
    Return sample from provided distribution. Must be named.
    """
    assert isinstance(dist, pyro.distributions.Distribution)
    return dist()


def observe(name, dist, obs):
    raise NotImplementedError(
        "Observe has been used outside of a normalizing context.")


def map_data(name, fn, data, batch_size=None):
    """
    top-level map_data: like map(fn, enumerate(data)), but with a name
    Assumes fn takes two arguments: ind, val
    infer algs (eg VI) that do minibatches should overide this.
    """
    if isinstance(data, torch.Tensor) or isinstance(data, Variable):
        return fn(Variable(torch.arange(0, data.size(0))), data)
    else:
        # list for py3 compatibility
        return list(map(lambda ix: fn(*ix), enumerate(data)))

# hand off behavior to poutine if necessary?
# for now default calls out to pyro.param -- which is handled by poutine


def module(pyro_name, nn_obj):  # :, *args, **kwargs):
    assert hasattr(nn_obj, "parameters")

    # cannot contain our special modifier marker
    assert mod_div not in pyro_name

    if isclass(nn_obj):
        raise NotImplementedError("Not yet supporting class constructor")

    # for now, we simply loop through parameters every time and marke then
    for param in nn_obj.parameters():

        # is param a variable? Save variable inside param store
        if isinstance(param, Variable):

            # mark the object
            pyro.param(module_name(pyro_name, param), param)

    # send back object for calling forward
    # if we want to intercept somewhere else we can
    return nn_obj
