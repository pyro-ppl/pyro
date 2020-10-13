# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from inspect import isclass

import pyro.distributions as dist
import pyro.infer as infer
import pyro.poutine as poutine
from pyro.params import param_with_module_name
from pyro.poutine.plate_messenger import PlateMessenger
from pyro.poutine.runtime import _MODULE_NAMESPACE_DIVIDER, _PYRO_PARAM_STORE, am_i_wrapped, apply_stack, effectful
from pyro.poutine.subsample_messenger import SubsampleMessenger
from pyro.util import deep_getattr, set_rng_seed  # noqa: F401


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


_param = effectful(_PYRO_PARAM_STORE.get_param, type="param")


def param(name, *args, **kwargs):
    """
    Saves the variable as a parameter in the param store.
    To interact with the param store or write to disk,
    see `Parameters <parameters.html>`_.

    :param str name: name of parameter
    :param init_tensor: initial tensor or lazy callable that returns a tensor.
        For large tensors, it may be cheaper to write e.g.
        ``lambda: torch.randn(100000)``, which will only be evaluated on the
        initial statement.
    :type init_tensor: torch.Tensor or callable
    :param constraint: torch constraint, defaults to ``constraints.real``.
    :type constraint: torch.distributions.constraints.Constraint
    :param int event_dim: (optional) number of rightmost dimensions unrelated
        to baching. Dimension to the left of this will be considered batch
        dimensions; if the param statement is inside a subsampled plate, then
        corresponding batch dimensions of the parameter will be correspondingly
        subsampled. If unspecified, all dimensions will be considered event
        dims and no subsampling will be performed.
    :returns: parameter
    :rtype: torch.Tensor
    """
    kwargs["name"] = name
    return _param(name, *args, **kwargs)


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
    :param dict infer: Optional dictionary of inference parameters specified
        in kwargs. See inference documentation for details.
    :returns: sample
    """
    obs = kwargs.pop("obs", None)
    infer = kwargs.pop("infer", {}).copy()
    # check if stack is empty
    # if stack empty, default behavior (defined here)
    if not am_i_wrapped():
        if obs is not None and not infer.get("_deterministic"):
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
            "infer": infer,
            "scale": 1.0,
            "mask": None,
            "cond_indep_stack": (),
            "done": False,
            "stop": False,
            "continuation": None
        }
        # handle observation
        if obs is not None:
            msg["value"] = obs
            msg["is_observed"] = True
        # apply the stack and return its return value
        apply_stack(msg)
        return msg["value"]


def factor(name, log_factor):
    """
    Factor statement to add arbitrary log probability factor to a
    probabilisitic model.

    :param str name: Name of the trivial sample
    :param torch.Tensor log_factor: A possibly batched log probability factor.
    """
    unit_dist = dist.Unit(log_factor)
    unit_value = unit_dist.sample()
    sample(name, unit_dist, obs=unit_value, infer={"is_auxiliary": True})


def deterministic(name, value, event_dim=None):
    """
    EXPERIMENTAL Deterministic statement to add a :class:`~pyro.distributions.Delta`
    site with name `name` and value `value` to the trace. This is useful when
    we want to record values which are completely determined by their parents.
    For example::

        x = sample("x", dist.Normal(0, 1))
        x2 = deterministic("x2", x ** 2)

    .. note:: The site does not affect the model density. This currently converts
        to a :func:`sample` statement, but may change in the future.

    :param str name: Name of the site.
    :param torch.Tensor value: Value of the site.
    :param int event_dim: Optional event dimension, defaults to `value.ndim`.
    """
    event_dim = value.ndim if event_dim is None else event_dim
    return sample(name, dist.Delta(value, event_dim=event_dim).mask(False),
                  obs=value, infer={"_deterministic": True})


@effectful(type="subsample")
def subsample(data, event_dim):
    """
    EXPERIMENTAL Subsampling statement to subsample data based on enclosing
    :class:`~pyro.primitives.plate` s.

    This is typically called on arguments to ``model()`` when subsampling is
    performed automatically by :class:`~pyro.primitives.plate` s by passing
    either the ``subsample`` or ``subsample_size`` kwarg. For example the
    following are equivalent::

        # Version 1. using indexing
        def model(data):
            with pyro.plate("data", len(data), subsample_size=10, dim=-data.dim()) as ind:
                data = data[ind]
                # ...

        # Version 2. using pyro.subsample()
        def model(data):
            with pyro.plate("data", len(data), subsample_size=10, dim=-data.dim()):
                data = pyro.subsample(data, event_dim=0)
                # ...

    :param data: A tensor of batched data.
    :type data: ~torch.Tensor
    :param int event_dim: The event dimension of the data tensor. Dimensions to
        the left are considered batch dimensions.
    :returns: A subsampled version of ``data``
    :rtype: ~torch.Tensor
    """
    assert isinstance(event_dim, int) and event_dim >= 0
    return data  # May be intercepted by SubsampleMessenger.


class plate(PlateMessenger):
    """
    Construct for conditionally independent sequences of variables.

    ``plate`` can be used either sequentially as a generator or in parallel as
    a context manager (formerly ``irange`` and ``iarange``, respectively).

    Sequential :class:`plate` is similar to :py:func:`range` in that it generates
    a sequence of values.

    Vectorized :class:`plate` is similar to :func:`torch.arange` in that it
    yields an array of indices by which other tensors can be indexed.
    :class:`plate` differs from :func:`torch.arange` in that it also informs
    inference algorithms that the variables being indexed are conditionally
    independent.  To do this, :class:`plate` is a provided as context manager
    rather than a function, and users must guarantee that all computation
    within an :class:`plate` context is conditionally independent::

        with plate("name", size) as ind:
            # ...do conditionally independent stuff with ind...

    Additionally, :class:`plate` can take advantage of the conditional
    independence assumptions by subsampling the indices and informing inference
    algorithms to scale various computed values. This is typically used to
    subsample minibatches of data::

        with plate("data", len(data), subsample_size=100) as ind:
            batch = data[ind]
            assert len(batch) == 100

    By default ``subsample_size=False`` and this simply yields a
    ``torch.arange(0, size)``. If ``0 < subsample_size <= size`` this yields a
    single random batch of indices of size ``subsample_size`` and scales all
    log likelihood terms by ``size/batch_size``, within this context.

    .. warning::  This is only correct if all computation is conditionally
        independent within the context.

    :param str name: A unique name to help inference algorithms match
        :class:`plate` sites between models and guides.
    :param int size: Optional size of the collection being subsampled
        (like `stop` in builtin `range`).
    :param int subsample_size: Size of minibatches used in subsampling.
        Defaults to `size`.
    :param subsample: Optional custom subsample for user-defined subsampling
        schemes. If specified, then `subsample_size` will be set to
        `len(subsample)`.
    :type subsample: Anything supporting `len()`.
    :param int dim: An optional dimension to use for this independence index.
        If specified, ``dim`` should be negative, i.e. should index from the
        right. If not specified, ``dim`` is set to the rightmost dim that is
        left of all enclosing ``plate`` contexts.
    :param bool use_cuda: DEPRECATED, use the `device` arg instead.
        Optional bool specifying whether to use cuda tensors for `subsample`
        and `log_prob`. Defaults to ``torch.Tensor.is_cuda``.
    :param str device: Optional keyword specifying which device to place
        the results of `subsample` and `log_prob` on. By default, results
        are placed on the same device as the default tensor.
    :return: A reusabe context manager yielding a single 1-dimensional
        :class:`torch.Tensor` of indices.

    Examples:

        .. doctest::
           :hide:

           >>> loc, scale = torch.tensor(0.), torch.tensor(1.)
           >>> data = torch.randn(100)
           >>> z = dist.Bernoulli(0.5).sample((100,))

        >>> # This version declares sequential independence and subsamples data:
        >>> for i in plate('data', 100, subsample_size=10):
        ...     if z[i]:  # Control flow in this example prevents vectorization.
        ...         obs = sample('obs_{}'.format(i), dist.Normal(loc, scale), obs=data[i])

        >>> # This version declares vectorized independence:
        >>> with plate('data'):
        ...     obs = sample('obs', dist.Normal(loc, scale), obs=data)

        >>> # This version subsamples data in vectorized way:
        >>> with plate('data', 100, subsample_size=10) as ind:
        ...     obs = sample('obs', dist.Normal(loc, scale), obs=data[ind])

        >>> # This wraps a user-defined subsampling method for use in pyro:
        >>> ind = torch.randint(0, 100, (10,)).long() # custom subsample
        >>> with plate('data', 100, subsample=ind):
        ...     obs = sample('obs', dist.Normal(loc, scale), obs=data[ind])

        >>> # This reuses two different independence contexts.
        >>> x_axis = plate('outer', 320, dim=-1)
        >>> y_axis = plate('inner', 200, dim=-2)
        >>> with x_axis:
        ...     x_noise = sample("x_noise", dist.Normal(loc, scale))
        ...     assert x_noise.shape == (320,)
        >>> with y_axis:
        ...     y_noise = sample("y_noise", dist.Normal(loc, scale))
        ...     assert y_noise.shape == (200, 1)
        >>> with x_axis, y_axis:
        ...     xy_noise = sample("xy_noise", dist.Normal(loc, scale))
        ...     assert xy_noise.shape == (200, 320)

    See `SVI Part II <http://pyro.ai/examples/svi_part_ii.html>`_ for an
    extended discussion.
    """
    pass


class iarange(plate):
    def __init__(self, *args, **kwargs):
        warnings.warn("pyro.iarange is deprecated; use pyro.plate instead", DeprecationWarning)
        super().__init__(*args, **kwargs)


class irange(SubsampleMessenger):
    def __init__(self, *args, **kwargs):
        warnings.warn("pyro.irange is deprecated; use pyro.plate instead", DeprecationWarning)
        super().__init__(*args, **kwargs)


@contextmanager
def plate_stack(prefix, sizes, rightmost_dim=-1):
    """
    Create a contiguous stack of :class:`plate` s with dimensions::

        rightmost_dim - len(sizes), ..., rightmost_dim

    :param str prefix: Name prefix for plates.
    :param iterable sizes: An iterable of plate sizes.
    :param int rightmost_dim: The rightmost dim, counting from the right.
    """
    assert rightmost_dim < 0
    with ExitStack() as stack:
        for i, size in enumerate(reversed(sizes)):
            plate_i = plate("{}_{}".format(prefix, i), size, dim=rightmost_dim - i)
            stack.enter_context(plate_i)
        yield


def module(name, nn_module, update_module_params=False):
    """
    Takes a torch.nn.Module and registers its parameters with the ParamStore.
    In conjunction with the ParamStore save() and load() functionality, this
    allows the user to save and load modules.

    :param name: name of module
    :type name: str
    :param nn_module: the module to be registered with Pyro
    :type nn_module: torch.nn.Module
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
        if param_value.requires_grad:
            # register the parameter in the module with pyro
            # this only does something substantive if the parameter hasn't been seen before
            full_param_name = param_with_module_name(name, param_name)
            returned_param = param(full_param_name, param_value)

            if param_value._cdata != returned_param._cdata:
                target_state_dict[param_name] = returned_param
        else:
            warnings.warn("{} was not registered in the param store because".format(param_name) +
                          " requires_grad=False")

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
    r"""
    .. warning::
        The `random_module` primitive is deprecated, and will be removed
        in a future release. Use :class:`~pyro.nn.module.PyroModule` instead to
        to create Bayesian modules from :class:`torch.nn.Module` instances.
        See the `Bayesian Regression tutorial <http://pyro.ai/examples/bayesian_regression.html>`_
        for an example.


    Places a prior over the parameters of the module `nn_module`.
    Returns a distribution (callable) over `nn.Module`\s, which
    upon calling returns a sampled `nn.Module`.

    :param name: name of pyro module
    :type name: str
    :param nn_module: the module to be registered with pyro
    :type nn_module: torch.nn.Module
    :param prior: pyro distribution, stochastic function, or python dict with parameter names
                  as keys and respective distributions/stochastic functions as values.
    :returns: a callable which returns a sampled module
    """
    warnings.warn("The `random_module` primitive is deprecated, and will be removed "
                  "in a future release. Use `pyro.nn.Module` to create Bayesian "
                  "modules from `torch.nn.Module` instances.", FutureWarning)

    assert hasattr(nn_module, "parameters"), "Module is not a NN module."
    # register params in param store
    lifted_fn = poutine.lift(module, prior=prior)

    def _fn():
        nn_copy = copy.deepcopy(nn_module)
        # update_module_params must be True or the lifted module will not update local params
        return lifted_fn(name, nn_copy, update_module_params=True, *args, **kwargs)
    return _fn


@effectful(type="barrier")
def barrier(data):
    """
    EXPERIMENTAL Ensures all values in ``data`` are ground, rather than lazy
    funsor values. This is useful in combination with
    :func:`pyro.poutine.collapse`.
    """
    return data


def enable_validation(is_validate=True):
    """
    Enable or disable validation checks in Pyro. Validation checks provide
    useful warnings and errors, e.g. NaN checks, validating distribution
    arguments and support values, etc. which is useful for debugging.
    Since some of these checks may be expensive, we recommend turning
    this off for mature models.

    :param bool is_validate: (optional; defaults to True) whether to
        enable validation checks.
    """
    dist.enable_validation(is_validate)
    infer.enable_validation(is_validate)
    poutine.enable_validation(is_validate)


@contextmanager
def validation_enabled(is_validate=True):
    """
    Context manager that is useful when temporarily enabling/disabling
    validation checks.

    :param bool is_validate: (optional; defaults to True) temporary
        validation check override.
    """
    infer_validation_status = infer.is_validation_enabled()
    distribution_validation_status = dist.is_validation_enabled()
    poutine_validation_status = poutine.is_validation_enabled()
    try:
        enable_validation(is_validate)
        yield
    finally:
        dist.enable_validation(distribution_validation_status)
        infer.enable_validation(infer_validation_status)
        poutine.enable_validation(poutine_validation_status)
