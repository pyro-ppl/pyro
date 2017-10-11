from __future__ import division

import warnings
import contextlib
from inspect import isclass

import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine

from pyro.params import param_with_module_name
from pyro.params.param_store import ParamStoreDict
from pyro.poutine import LambdaPoutine, condition, do  # noqa: F401
from pyro.util import zeros, ones, set_rng_seed, apply_stack  # noqa: F401

# global map of params for now
_param_store = ParamStoreDict()

# used to create fully-formed param names, e.g. mymodule$$$mysubmodule.weight
_MODULE_NAMESPACE_DIVIDER = "$$$"


def get_param_store():
    """
    Returns the param store
    """
    return _param_store


def clear_param_store():
    """
    Clears the param store
    """
    return _param_store.clear()


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

_PYRO_STACK = []


def sample(name, fn, *args, **kwargs):
    """
    :param name: name of sample
    :param fn: distribution class or function
    :param obs: observed datum (optional; should only be used in context of inference)
        optionally specified in kwargs
    :returns: sample

    Samples from the distribution and registers it in the trace data structure.
    """
    obs = kwargs.pop("obs", None)
    # check if stack is empty
    # if stack empty, default behavior (defined here)
    if len(_PYRO_STACK) == 0:
        if obs is not None:
            warnings.warn("trying to observe a value outside of inference at " + name,
                          warnings.RuntimeWarning)
        return fn(*args, **kwargs)
    # if stack not empty, apply everything in the stack?
    else:
        # initialize data structure to pass up/down the stack
        if obs is None:
            msg_type = "sample"
        else:
            msg_type = "observe"
        msg = {
            "type": msg_type,
            "name": name,
            "fn": fn,
            "obs": obs,
            "args": args,
            "kwargs": kwargs,
            "value": None,
            "scale": 1.0,
            "map_data_stack": [],
            "done": False,
            "stop": False,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["value"]


def observe(name, fn, obs, *args, **kwargs):
    """
    :param name: name of observation
    :param fn: distribution class or function
    :param obs: observed datum
    :returns: sample

    Alias of pyro.sample.

    Only should be used in the context of inference.
    Calculates the score of the sample and registers
    it in the trace data structure.
    """
    kwargs.update({"obs": obs})
    return sample(name, fn, *args, **kwargs)


@contextlib.contextmanager
def iarange(name, size, subsample_size=0):
    """
    Context manager for ranges indexing iid variables, optionally subsampling.

    WARNING: Subsampling is only correct if all computation is iid within the context.

    By default `subsample_size=False` and this simply yields a `torch.arange(0, size)`.
    If `0<subsample_size<=size` this yields a single random batch of size
    `subsample_size` and scales all log likelihood terms by `size/batch_size`, within
    this context.

    :param str name: A name that will be used for this site in a Trace.
    :param int size: The size of the collection being subsampled (like `stop` in builtin `range`).
    :param int subsample_size: Size of minibatches used in subsampling. Defaults to `size` if set to 0.
    :return: A context manager yielding a single 1-dimensional `torch.Tensor` of indices.

    Examples::

        # This version is vectorized:
        >>> with iarange('data', 100, subsample_size=10) as batch:
                observe('obs', normal, data.index_select(0, batch), mu, sigma)
        # This version manually iterates through the batch to deal with control flow.
        >>> with iarange('data', 100, subsample_size=10) as batch:
                for i in batch:
                    if z[i]:  # Prevents vectorization.
                        observe('obs_{}'.format(i), normal, data[i], mu, sigma)
    """
    if subsample_size == 0 or subsample_size >= size:
        subsample_size = size
    if subsample_size == size:
        # If not subsampling, there is no need to scale and we can ignore the _PYRO_STACK.
        yield Variable(torch.LongTensor(list(range(size))))
        return

    subsample = Variable(torch.randperm(size)[0:subsample_size])
    if len(_PYRO_STACK) == 0:
        yield subsample
    else:
        # Wrap computation in a scaling context.
        scale = size / subsample_size
        with LambdaPoutine(None, name, scale, 'tensor', 0, subsample_size):
            yield subsample


def irange(name, size, subsample_size=0):
    """
    Non-vectorized version of iarange.

    Examples::

        >>> for i in irange('data', 100, subsample_size=10):
                if z[i]:  # Prevents vectorization.
                    observe('obs_{}'.format(i), normal, data[i], mu, sigma)
    """
    with iarange(name, size, subsample_size) as batch:
        # Wrap computation in an independence context.
        indep_context = LambdaPoutine(None, name, 1.0, 'list', 0, subsample_size)
        for i in batch.data:
            with indep_context:
                yield i


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
    if isinstance(data, (torch.Tensor, Variable)):
        size = data.size(batch_dim)
        with iarange(name, size, batch_size) as batch:
            return fn(batch, data.index_select(batch_dim, batch))
    else:
        size = len(data)
        return [fn(i, data[i]) for i in irange(name, size, batch_size)]


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
            "map_data_stack": [],
            "value": None,
            "done": False,
            "stop": False,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["value"]


def module(pyro_name, nn_obj, tags="default"):
    """
    :param pyro_name: name of module
    :type pyro_name: str
    :param nn_obj: the module to be registered with pyro
    :type nn_obj: torch.nn.Module
    :param tags: optional; tags to associate with any parameters inside the module
    :type tags: string or iterable of strings
    :returns: torch.nn.Module

    Takes a torch.nn.Module and registers its parameters with the param store.
    In conjunction with the param store save() and load() functionality, this
    allows the user to save and load modules.
    """
    assert hasattr(nn_obj, "parameters"), "module has no parameters"
    assert _MODULE_NAMESPACE_DIVIDER not in pyro_name, "improper module name, since contains %s" %\
        _MODULE_NAMESPACE_DIVIDER

    if isclass(nn_obj):
        raise NotImplementedError("Not yet supporting class constructor")

    state_dict = {}
    for param_name, param in nn_obj.named_parameters():
        state_dict[param_name] = pyro.param(param_with_module_name(pyro_name, param_name), param,
                                            tags=tags)

    current_nn_state = nn_obj.state_dict()
    for name, param in state_dict.items():
        if name not in current_nn_state:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if isinstance(param, Variable):
            # backwards compatibility for serialized parameters
            param = param.data

        # only copy if the param has actually changed
        # Note: apart from the following line, the rest of this code
        # logic is borrowed from torch.nn.Module.load_state_dict
        if id(param) != id(current_nn_state[name]):
            current_nn_state[name].copy_(param)

    # the KeyError below may not be correct; see github issue
    missing = set(current_nn_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    nn_obj.load_state_dict(current_nn_state)
    return nn_obj


def random_module(name, nn_module, prior, *args, **kwargs):
    """
    :param name: name of pyro module
    :param nn_module: pytorch nn module
    :param prior: prior distribution or iterable over distributions
    :returns: a callable which returns a sampled module

    Places a prior over the parameters of the nn module
    """
    assert hasattr(nn_module, "parameters"), "Module is not a NN module."
    # register params in param store
    lifted_fn = poutine.lift(pyro.module, prior, *args, **kwargs)
    return lambda: lifted_fn(name, nn_module)
