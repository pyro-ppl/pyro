from __future__ import absolute_import, division, print_function

import contextlib
import copy
import warnings
from collections import OrderedDict
from inspect import isclass

import torch
from torch.autograd import Variable

import pyro.poutine as poutine
from pyro.distributions.distribution import Distribution
from pyro.params import _MODULE_NAMESPACE_DIVIDER, _PYRO_PARAM_STORE, param_with_module_name
from pyro.poutine import _PYRO_STACK, condition, do  # noqa: F401
from pyro.util import apply_stack, deep_getattr, get_tensor_data, ones, set_rng_seed, zeros  # noqa: F401

__version__ = '0.1.2'


def get_param_store():
    """
    Returns the ParamStore
    """
    return _PYRO_PARAM_STORE


def clear_param_store():
    """
    Clears the ParamStore. This is especially useful if you're working in a REPL.
    """
    return _PYRO_PARAM_STORE.clear()


def sample(name, fn, *args, **kwargs):
    """
    Calls the stochastic function `fn` with additional side-effects depending
    on `name` and the enclosing context (e.g. an inference algorithm).
    See `Intro I <http://pyro.ai/examples/intro_part_i.html>`_ and
    `Intro II <http://pyro.ai/examples/intro_part_ii.html>`_ for a discussion.

    :param name: name of sample
    :param fn: distribution class or function
    :param obs: observed datum (optional; should only be used in context of
        inference) optionally specified in kwargs
    :param dict baseline: Optional dictionary of baseline parameters specified
        in kwargs. See inference documentation for details.
    :returns: sample
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
            "cond_indep_stack": [],
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
    Alias of `pyro.sample(name, fn, *args, obs=obs, **kwargs)`.

    :param name: name of observation
    :param fn: distribution class or function
    :param obs: observed datum
    :returns: sample
    """
    kwargs.update({"obs": obs})
    return sample(name, fn, *args, **kwargs)


class _Subsample(Distribution):
    """
    Randomly select a subsample of a range of indices.

    Internal use only. This should only be used by `iarange`.
    """

    def __init__(self, size, subsample_size, use_cuda=None):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        :param bool use_cuda: whether to use cuda tensors
        """
        self.size = size
        self.subsample_size = subsample_size
        self.use_cuda = torch.Tensor.is_cuda if use_cuda is None else use_cuda

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


def _subsample(name, size=None, subsample_size=None, subsample=None, use_cuda=None):
    """
    Helper function for iarange and irange. See their docstrings for details.
    """
    if size is None:
        assert subsample_size is None
        assert subsample is None
        size = 1
        subsample_size = 1
    elif subsample is None:
        names = [name]
        names += [str(f.counter) for f in _PYRO_STACK if isinstance(f, poutine.IndepPoutine)]
        subsample = sample("_".join(names), _Subsample(size, subsample_size, use_cuda))

    if subsample_size is None:
        subsample_size = len(subsample)
    elif subsample is not None and subsample_size != len(subsample):
        raise ValueError("subsample_size does not match len(subsample), {} vs {}.".format(
            subsample_size, len(subsample)) +
            " Did you accidentally use different subsample_size in the model and guide?")

    scale = size / subsample_size
    return subsample, scale


@contextlib.contextmanager
def iarange(name, size=None, subsample_size=None, subsample=None, use_cuda=None):
    """
    Context manager for conditionally independent ranges of variables.

    ``iarange`` is similar to ``torch.arange`` in that it yields an array
    of indices by which other tensors can be indexed. ``iarange`` differs from
    ``torch.arange`` in that it also informs inference algorithms that the
    variables being indexed are conditionally independent. To do this,
    ``iarange`` is a provided as context manager rather than a function, and
    users must guarantee that all computation within an ``iarange`` context
    is conditionally independent::

        with iarange("name", size) as ind:
            # ...do conditionally independent stuff with ind...

    Additionally, ``iarange`` can take advantage of the conditional
    independence assumptions by subsampling the indices and informing inference
    algorithms to scale various computed values. This is typically used to
    subsample minibatches of data::

        with iarange("data", len(data), subsample_size=100) as ind:
            batch = data[ind]
            assert len(batch) == 100

    By default ``subsample_size=False`` and this simply yields a
    ``torch.arange(0, size)``. If ``0 < subsample_size <= size`` this yields a
    single random batch of indices of size ``subsample_size`` and scales all
    log likelihood terms by ``size/batch_size``, within this context.

    .. warning::  This is only correct if all computation is conditionally
        independent within the context.

    :param str name: A unique name to help inference algorithms match
        ``iarange`` sites between models and guides.
    :param int size: Optional size of the collection being subsampled
        (like `stop` in builtin `range`).
    :param int subsample_size: Size of minibatches used in subsampling.
        Defaults to `size`.
    :param subsample: Optional custom subsample for user-defined subsampling
        schemes. If specified, then `subsample_size` will be set to
        `len(subsample)`.
    :type subsample: Anything supporting `len()`.
    :param bool use_cuda: Optional bool specifying whether to use cuda tensors
        for `subsample` and `log_pdf`. Defaults to `torch.Tensor.is_cuda`.
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

    See `SVI Part II <http://pyro.ai/examples/svi_part_ii.html>`_ for an
    extended discussion.
    """
    subsample, scale = _subsample(name, size, subsample_size, subsample, use_cuda)
    if len(_PYRO_STACK) == 0:
        yield subsample
    else:
        with poutine.scale(None, scale):
            with poutine.indep(None, name, vectorized=True):
                yield subsample


def irange(name, size, subsample_size=None, subsample=None, use_cuda=None):
    """
    Non-vectorized version of ``iarange``. See ``iarange`` for details.

    :param str name: A name that will be used for this site in a Trace.
    :param int size: The size of the collection being subsampled (like ``stop``
        in builtin ``range``).
    :param int subsample_size: Size of minibatches used in subsampling.
        Defaults to ``size``.
    :param subsample: Optional custom subsample for user-defined subsampling
        schemes. If specified, then ``subsample_size`` will be set to
        ``len(subsample)``.
    :type subsample: Anything supporting ``len()``.
    :param bool use_cuda: Optional bool specifying whether to use cuda tensors
        for internal ``log_pdf`` computations. Defaults to
        ``torch.Tensor.is_cuda``.
    :return: A generator yielding a sequence of integers.

    Examples::

        >>> for i in irange('data', 100, subsample_size=10):
                if z[i]:  # Prevents vectorization.
                    observe('obs_{}'.format(i), normal, data[i], mu, sigma)

    See `SVI Part II <http://pyro.ai/examples/svi_part_ii.html>`_ for an extended discussion.
    """
    subsample, scale = _subsample(name, size, subsample_size, subsample, use_cuda)
    if isinstance(subsample, Variable):
        subsample = subsample.data
    if len(_PYRO_STACK) == 0:
        for i in subsample:
            yield i
    else:
        indep_context = poutine.indep(None, name, vectorized=False)
        with poutine.scale(None, scale):
            for i in subsample:
                with indep_context:
                    yield i


def map_data(name, data, fn, batch_size=None, batch_dim=0, use_cuda=None):
    """
    Data subsampling with the important property that all the data are conditionally independent.

    With default values of `batch_size` and `batch_dim`, `map_data` behaves like `map`.
    More precisely, `map_data('foo', data, fn)` is equivalent to `[fn(i, x) for i, x in enumerate(data)]`.

    :param str name: named argument
    :param data: data to subsample
    :param callable fn: a function taking `(index, datum)` pairs, where `dataum = data[index]`
    :param int batch_size: number of samples per batch, or zero for the entire dataset
    :param int batch_dim: dimension to subsample for tensor inputs
    :param bool use_cuda: Optional bool specifying whether to use cuda tensors
        for `log_pdf`. Defaults to `torch.Tensor.is_cuda`.
    :return: a list of values returned by `fn`
    """

    use_cuda = use_cuda or getattr(data, 'is_cuda', None)
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
    Saves the variable as a parameter in the param store.
    To interact with the param store or write to disk,
    see `Parameters <parameters.html>`_.

    :param name: name of parameter
    :returns: parameter
    """
    if len(_PYRO_STACK) == 0:
        return _PYRO_PARAM_STORE.get_param(name, *args, **kwargs)
    else:
        msg = {
            "type": "param",
            "name": name,
            "args": args,
            "kwargs": kwargs,
            "scale": 1.0,
            "cond_indep_stack": [],
            "value": None,
            "done": False,
            "stop": False,
        }
        # apply the stack and return its return value
        out_msg = apply_stack(msg)
        return out_msg["value"]


def module(name, nn_module, tags="default", update_module_params=False):
    """
    Takes a torch.nn.Module and registers its parameters with the ParamStore.
    In conjunction with the ParamStore save() and load() functionality, this
    allows the user to save and load modules.

    :param name: name of module
    :type name: str
    :param nn_module: the module to be registered with Pyro
    :type nn_module: torch.nn.Module
    :param tags: optional; tags to associate with any parameters inside the module
    :type tags: string or iterable of strings
    :param update_module_params: determines whether Parameters
                                 in the PyTorch module get overridden with the values found in the
                                 ParamStore (if any). Defaults to `False`
    :type load_from_param_store: bool
    :returns: torch.nn.Module
    """
    assert hasattr(nn_module, "parameters"), "module has no parameters"
    assert _MODULE_NAMESPACE_DIVIDER not in name, "improper module name, since contains %s" %\
        _MODULE_NAMESPACE_DIVIDER

    if isclass(nn_module):
        raise NotImplementedError("pyro.module does not support class constructors for " +
                                  "the argument nn_module")

    target_state_dict = OrderedDict()

    for param_name, param_value in nn_module.named_parameters():
        # register the parameter in the module with pyro
        # this only does something substantive if the parameter hasn't been seen before
        full_param_name = param_with_module_name(name, param_name)
        returned_param = param(full_param_name, param_value, tags=tags)

        if get_tensor_data(param_value)._cdata != get_tensor_data(returned_param)._cdata:
            target_state_dict[param_name] = returned_param

    if target_state_dict and update_module_params:
        # WARNING: this is very dangerous. better method?
        for _name, _param in nn_module.named_parameters():
            is_param = False
            name_arr = _name.rsplit('.', 1)
            if len(name_arr) > 1:
                mod_name, param_name = name_arr[0], name_arr[1]
            else:
                is_param = True
                mod_name = _name
            if _name in target_state_dict.keys():
                if not is_param:
                    deep_getattr(nn_module, mod_name)._parameters[param_name] = target_state_dict[_name]
                else:
                    nn_module._parameters[mod_name] = target_state_dict[_name]

    return nn_module


def random_module(name, nn_module, prior, *args, **kwargs):
    """
    Places a prior over the parameters of the module `nn_module`.
    Returns a distribution (callable) over `nn.Module`s, which
    upon calling returns a sampled `nn.Module`.

    See the `Bayesian Regression tutorial <http://pyro.ai/examples/bayesian_regression.html>`_
    for an example.

    :param name: name of pyro module
    :type name: str
    :param nn_module: the module to be registered with pyro
    :type nn_module: torch.nn.Module
    :param prior: pyro distribution, stochastic function, or python dict with parameter names
                  as keys and respective distributions/stochastic functions as values.
    :returns: a callable which returns a sampled module
    """
    assert hasattr(nn_module, "parameters"), "Module is not a NN module."
    # register params in param store
    lifted_fn = poutine.lift(module, prior)

    def _fn():
        nn_copy = copy.deepcopy(nn_module)
        # update_module_params must be True or the lifted module will not update local params
        return lifted_fn(name, nn_copy, update_module_params=True, *args, **kwargs)
    return _fn
