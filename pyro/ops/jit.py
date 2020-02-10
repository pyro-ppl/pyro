# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import warnings
import weakref

import torch

import pyro
import pyro.poutine as poutine
from pyro.util import ignore_jit_warnings, optional, timed


def _hash(value, allow_id):
    try:
        hash(value)
        return value
    except TypeError as e:
        if isinstance(value, list):
            return tuple(_hash(x, allow_id) for x in value)
        elif isinstance(value, dict):
            return tuple(sorted((_hash(x, allow_id), _hash(y, allow_id)) for x, y in value.items()))
        elif isinstance(value, set):
            return frozenset(_hash(x, allow_id) for x in value)
        elif isinstance(value, argparse.Namespace):
            return str(value)
        elif allow_id:
            return id(value)
        raise e


def _hashable_args_kwargs(args, kwargs):
    items = sorted(kwargs.items())
    hashable_kwargs = tuple((key, _hash(value, False)) for key, value in items)
    try:
        hash(hashable_kwargs)
    except TypeError:
        warnings.warn("Failed to hash kwargs; attempting to hash by id.")
        hashable_kwargs = tuple((key, _hash(value, True)) for key, value in items)
    return len(args), hashable_kwargs


class CompiledFunction:
    """
    Output type of :func:`pyro.ops.jit.trace`.

    Wrapper around the output of :func:`torch.jit.trace`
    that handles parameter plumbing.

    The actual PyTorch compilation artifact is stored in :attr:`compiled`.
    Call diagnostic methods on this attribute.
    """
    def __init__(self, fn, ignore_warnings=False, jit_options=None):
        self.fn = fn
        self.compiled = {}  # len(args) -> callable
        self.ignore_warnings = ignore_warnings
        self.jit_options = {} if jit_options is None else jit_options
        self.jit_options.setdefault('check_trace', False)
        self.compile_time = None
        self._param_names = None

    def __call__(self, *args, **kwargs):
        key = _hashable_args_kwargs(args, kwargs)

        # if first time
        if key not in self.compiled:
            # param capture
            with poutine.block():
                with poutine.trace(param_only=True) as first_param_capture:
                    self.fn(*args, **kwargs)

            self._param_names = list(set(first_param_capture.trace.nodes.keys()))
            unconstrained_params = tuple(pyro.param(name).unconstrained()
                                         for name in self._param_names)
            params_and_args = unconstrained_params + args
            weakself = weakref.ref(self)

            def compiled(*params_and_args):
                self = weakself()
                unconstrained_params = params_and_args[:len(self._param_names)]
                args = params_and_args[len(self._param_names):]
                constrained_params = {}
                for name, unconstrained_param in zip(self._param_names, unconstrained_params):
                    constrained_param = pyro.param(name)  # assume param has been initialized
                    assert constrained_param.unconstrained() is unconstrained_param
                    constrained_params[name] = constrained_param
                return poutine.replay(self.fn, params=constrained_params)(*args, **kwargs)

            if self.ignore_warnings:
                compiled = ignore_jit_warnings()(compiled)
            with pyro.validation_enabled(False):
                time_compilation = self.jit_options.pop("time_compilation", False)
                with optional(timed(), time_compilation) as t:
                    self.compiled[key] = torch.jit.trace(compiled, params_and_args, **self.jit_options)
                if time_compilation:
                    self.compile_time = t.elapsed
        else:
            unconstrained_params = [pyro.param(name).unconstrained()
                                    for name in self._param_names]
            params_and_args = unconstrained_params + list(args)

        with poutine.block(hide=self._param_names):
            with poutine.trace(param_only=True) as param_capture:
                ret = self.compiled[key](*params_and_args)

        for name in param_capture.trace.nodes.keys():
            if name not in self._param_names:
                raise NotImplementedError('pyro.ops.jit.trace assumes all params are created on '
                                          'first invocation, but found new param: {}'.format(name))

        return ret


def trace(fn=None, ignore_warnings=False, jit_options=None):
    """
    Lazy replacement for :func:`torch.jit.trace` that works with
    Pyro functions that call :func:`pyro.param`.

    The actual compilation artifact is stored in the ``compiled`` attribute of
    the output. Call diagnostic methods on this attribute.

    Example::

        def model(x):
            scale = pyro.param("scale", torch.tensor(0.5), constraint=constraints.positive)
            return pyro.sample("y", dist.Normal(x, scale))

        @pyro.ops.jit.trace
        def model_log_prob_fn(x, y):
            cond_model = pyro.condition(model, data={"y": y})
            tr = pyro.poutine.trace(cond_model).get_trace(x)
            return tr.log_prob_sum()

    :param callable fn: The function to be traced.
    :param bool ignore_warnins: Whether to ignore jit warnings.
    :param dict jit_options: Optional dict of options to pass to
        :func:`torch.jit.trace` , e.g. ``{"optimize": False}``.
    """
    if fn is None:
        return lambda fn: trace(fn, ignore_warnings=ignore_warnings, jit_options=jit_options)
    return CompiledFunction(fn, ignore_warnings=ignore_warnings, jit_options=jit_options)
