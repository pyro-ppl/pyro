from __future__ import division

import warnings
import contextlib
from inspect import isclass
from collections import OrderedDict

import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine

from pyro.distributions.distribution import Distribution
from pyro.params import param_with_module_name
from pyro.params.param_store import ParamStoreDict
from pyro.poutine import LambdaPoutine, condition, do  # noqa: F401
from pyro.util import (  # noqa: F401
                       zeros,
                       ones,
                       set_rng_seed,
                       apply_stack,
                       get_tensor_data,
                       deep_getattr
                       )

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
    :param obs: observed datum (optional; should only be used in context of
        inference) optionally specified in kwargs
    :param dict infer: Optional dictionary of inference parameters specified
        in kwargs. See inference documentation for details.
    :returns: sample

    Samples from the distribution and registers it in the trace data structure.
    """
    obs = kwargs.pop("obs", None)
    baseline = kwargs.pop("baseline", {})
    # check if stack is empty
    # if stack empty, default behavior (defined here)
    if len(_PYRO_STACK) == 0:
        if obs is not None:
            warnings.warn("trying to observe a value outside of inference at " + name,
                          RuntimeWarning)
            return obs
        return fn(*args, **kwargs)
    # if stack not empty, apply everything in the stack?
    else:
        # initialize data structure to pass up/down the stack
        msg = {
            "type": "sample",
            "name": name,
            "fn": fn,
            "is_observed": False,
            "args": args,
            "kwargs": kwargs,
            "value": None,
            "baseline": baseline,
            "scale": 1.0,
            "map_data_stack": [],
            "done": False,
            "stop": False,
        }
        # handle observation
        if obs is not None:
            msg["value"] = obs
            msg["is_observed"] = True
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


class _Subsample(Distribution):
    """
    Randomly select a subsample of a range of indices.

    Internal use only. This should only be used by `iarange`.
    """

    def __init__(self, size, subsample_size, use_cuda=False):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        """
        self.size = size
        self.subsample_size = subsample_size
        self.use_cuda = use_cuda

    def sample(self):
        """
        :returns: a random subsample of `range(size)`
        :rtype: torch.autograd.Variable of torch.LongTensor
        """
        subsample_size = self.subsample_size
        if subsample_size is None or subsample_size > self.size:
            subsample_size = self.size
        if subsample_size == self.size:
            result = Variable(torch.LongTensor(list(range(self.size))))
        else:
            result = Variable(torch.randperm(self.size)[:self.subsample_size])
        return result.cuda() if self.use_cuda else result

    def batch_log_pdf(self, x):
        # This is zero so that iarange can provide an unbiased estimate of
        # the non-subsampled batch_log_pdf.
        result = Variable(torch.zeros(1))
        return result.cuda() if self.use_cuda else result


@contextlib.contextmanager
def iarange(name, size=None, subsample_size=None, subsample=None, use_cuda=False):
    """
    Context manager for ranges indexing iid variables, optionally subsampling.

    WARNING: This is only correct if all computation is iid within the context.

    By default `subsample_size=False` and this simply yields a
    `torch.arange(0, size)`. If `0 < subsample_size <= size` this yields a
    single random batch of indices of size `subsample_size` and scales all log
    likelihood terms by `size/batch_size`, within this context.

    :param str name: A name that will be used for this site in a Trace.
    :param int size: Optional size of the collection being subsampled
        (like `stop` in builtin `range`).
    :param int subsample_size: Size of minibatches used in subsampling.
        Defaults to `size`.
    :param subsample: Optional custom subsample for user-defined subsampling
        schemes. If specified, then `subsample_size` will be set to
        `len(subsample)`.
    :type subsample: Anything supporting `len()`.
    :param bool use_cuda: Whether to use cuda tensors for `subsample` and
        `log_pdf`.
    :return: A context manager yielding a single 1-dimensional `torch.Tensor`
        of indices.

    Examples::

        # This version simply declares independence:
        >>> with iarange('data'):
                observe('obs', normal, data, mu, sigma)

        # This version subsamples data in vectorized way:
        >>> with iarange('data', 100, subsample_size=10) as ind:
                observe('obs', normal, data.index_select(0, ind), mu, sigma)

        # This wraps a user-defined subsampling method for use in pyro:
        >>> ind = my_custom_subsample
        >>> with iarange('data', 100, subsample=ind):
                observe('obs', normal, data.index_select(0, ind), mu, sigma)
    """
    if size is None:
        assert subsample_size is None
        assert subsample is None
        size = 1
        subsample_size = 1
    elif subsample is None:
        names = [name]
        names += [str(f.counter) for f in _PYRO_STACK if isinstance(f, poutine.LambdaPoutine)]
        subsample = sample("_".join(names), _Subsample(size, subsample_size, use_cuda))

    if subsample_size is None:
        subsample_size = len(subsample)
    elif subsample_size != len(subsample):
        raise ValueError("subsample_size does not match len(subsample), {} vs {}.".format(
            subsample_size, len(subsample)) +
            " Did you accidentally use different subsample_size in the model and guide?")

    if len(_PYRO_STACK) == 0:
        yield subsample
    else:
        # Wrap computation in a scaling context.
        scale = size / subsample_size
        with LambdaPoutine(None, name, scale, 'tensor', 0, subsample_size):
            yield subsample


def irange(name, size, subsample_size=None, subsample=None, use_cuda=False):
    """
    Non-vectorized version of `iarange`. See `iarange` for details.

    :param str name: A name that will be used for this site in a Trace.
    :param int size: The size of the collection being subsampled (like `stop` in builtin `range`).
    :param int subsample_size: Size of minibatches used in subsampling. Defaults to `size`.
    :param subsample: Optional custom subsample for user-defined subsampling schemes.
        If specified, then `subsample_size` will be set to `len(subsample)`.
    :type subsample: Anything supporting `len()`.
    :param bool use_cuda: Whether to use cuda tensors for `subsample` and `log_pdf`.
    :return: A context manager yielding a single 1-dimensional `torch.Tensor` of indices.

    Examples::

        >>> for i in irange('data', 100, subsample_size=10):
                if z[i]:  # Prevents vectorization.
                    observe('obs_{}'.format(i), normal, data[i], mu, sigma)
    """
    with iarange(name, size, subsample_size, subsample, use_cuda) as ind:
        if isinstance(ind, Variable):
            ind = ind.data
        if len(_PYRO_STACK) == 0:
            for i in ind:
                yield i
        else:
            # Wrap computation in an independence context.
            indep_context = LambdaPoutine(None, name, 1.0, 'list', 0, len(ind))
            for i in ind:
                with indep_context:
                    yield i


def map_data(name, data, fn, batch_size=None, batch_dim=0, use_cuda=False):
    """
    Data subsampling with the important property that all the data are conditionally independent.

    With default values of `batch_size` and `batch_dim`, `map_data` behaves like `map`.
    More precisely, `map_data('foo', data, fn)` is equivalent to `[fn(i, x) for i, x in enumerate(data)]`.

    :param str name: named argument
    :param data: data to subsample
    :param callable fn: a function taking `(index, datum)` pairs, where `dataum = data[index]`
    :param int batch_size: number of samples per batch, or zero for the entire dataset
    :param int batch_dim: dimension to subsample for tensor inputs
    :param bool use_cuda: Whether to use cuda tensors for `subsample` and `log_pdf`.
    :return: a list of values returned by `fn`
    """
    use_cuda = use_cuda or getattr(data, 'is_cuda', False)
    if isinstance(data, (torch.Tensor, Variable)):
        size = data.size(batch_dim)
        with iarange(name, size, batch_size, use_cuda=use_cuda) as batch:
            return fn(batch, data.index_select(batch_dim, batch))
    else:
        size = len(data)
        return [fn(i, data[i]) for i in irange(name, size, batch_size, use_cuda=use_cuda)]


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


def module(pyro_name, nn_obj, tags="default", update_module_params=False):
    """
    :param pyro_name: name of module
    :type pyro_name: str
    :param nn_obj: the module to be registered with pyro
    :type nn_obj: torch.nn.Module
    :param tags: optional; tags to associate with any parameters inside the module
    :type tags: string or iterable of strings
    :param update_module_params: flag to determine whether to overwrite parameters
                                 in the pytorch module with the values found in the
                                 paramstore. Defaults to `True`
    :type load_from_param_store: bool
    :returns: torch.nn.Module

    Takes a torch.nn.Module and registers its parameters with the param store.
    In conjunction with the param store save() and load() functionality, this
    allows the user to save and load modules.
    """
    assert hasattr(nn_obj, "parameters"), "module has no parameters"
    assert _MODULE_NAMESPACE_DIVIDER not in pyro_name, "improper module name, since contains %s" %\
        _MODULE_NAMESPACE_DIVIDER

    if isclass(nn_obj):
        raise NotImplementedError("pyro.module does not support class constructors for " +
                                  "the argument nn_obj")

    target_state_dict = OrderedDict()

    for param_name, param in nn_obj.named_parameters():
        # register the parameter in the module with pyro
        # this only does something substantive if the parameter hasn't been seen before
        full_param_name = param_with_module_name(pyro_name, param_name)
        returned_param = pyro.param(full_param_name, param, tags=tags)

        if get_tensor_data(param)._cdata != get_tensor_data(returned_param)._cdata:
            target_state_dict[param_name] = returned_param

    if target_state_dict and update_module_params:
        # WARNING: this is very dangerous. better method?
        for name, param in nn_obj.named_parameters():
            is_param = False
            name_arr = name.rsplit('.', 1)
            if len(name_arr) > 1:
                mod_name, param_name = name_arr[0], name_arr[1]
            else:
                is_param = True
                mod_name = name
            if name in target_state_dict.keys():
                if not is_param:
                    deep_getattr(nn_obj, mod_name)._parameters[param_name] = target_state_dict[name]
                else:
                    nn_obj._parameters[mod_name] = target_state_dict[name]

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
    lifted_fn = poutine.lift(pyro.module, prior)
    # update_module_params must be True or the lifted module will not update local params
    return lambda: lifted_fn(name, nn_module, update_module_params=True, *args, **kwargs)
