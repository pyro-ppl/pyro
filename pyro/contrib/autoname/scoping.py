# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
``pyro.contrib.autoname.scoping`` contains the implementation of
:func:`pyro.contrib.autoname.scope`, a tool for automatically appending
a semantically meaningful prefix to names of sample sites.
"""
import functools

from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful


class NameCountMessenger(Messenger):
    """
    ``NameCountMessenger`` is the implementation of :func:`pyro.contrib.autoname.name_count`
    """
    def __enter__(self):
        self._names = set()
        return super().__enter__()

    def _increment_name(self, name, label):
        while (name, label) in self._names:
            split_name = name.split("__")
            if "__" in name and split_name[-1].isdigit():
                counter = int(split_name[-1]) + 1
                name = "__".join(split_name[:-1] + [str(counter)])
            else:
                name = name + "__1"
        return name

    def _pyro_sample(self, msg):
        msg["name"] = self._increment_name(msg["name"], "sample")

    def _pyro_post_sample(self, msg):
        self._names.add((msg["name"], "sample"))

    def _pyro_post_scope(self, msg):
        self._names.add((msg["args"][0], "scope"))

    def _pyro_scope(self, msg):
        msg["args"] = (self._increment_name(msg["args"][0], "scope"),)


class ScopeMessenger(Messenger):
    """
    ``ScopeMessenger`` is the implementation of :func:`pyro.contrib.autoname.scope`
    """
    def __init__(self, prefix=None, inner=None):
        super().__init__()
        self.prefix = prefix
        self.inner = inner

    @staticmethod
    @effectful(type="scope")
    def _collect_scope(prefixed_scope):
        return prefixed_scope.split("/")[-1]

    def __enter__(self):
        if self.prefix is None:
            raise ValueError("no prefix was provided")
        if not self.inner:
            # to accomplish adding a counter to duplicate scopes,
            # we make ScopeMessenger.__enter__ effectful
            # so that the same mechanism that adds counters to sample names
            # can be used to add a counter to a scope name
            self.prefix = self._collect_scope(self.prefix)
        return super().__enter__()

    def __call__(self, fn):
        if self.prefix is None:
            self.prefix = fn.__code__.co_name  # fn.__name__

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with type(self)(prefix=self.prefix, inner=self.inner):
                return fn(*args, **kwargs)
        return _fn

    def _pyro_scope(self, msg):
        msg["args"] = ("{}/{}".format(self.prefix, msg["args"][0]),)

    def _pyro_sample(self, msg):
        msg["name"] = "{}/{}".format(self.prefix, msg["name"])


def scope(fn=None, prefix=None, inner=None):
    """
    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param prefix: a string to prepend to sample names (optional if ``fn`` is provided)
    :param inner: switch to determine where duplicate name counters appear
    :returns: ``fn`` decorated with a :class:`~pyro.contrib.autoname.scoping.ScopeMessenger`

    ``scope`` prepends a prefix followed by a ``/`` to the name at a Pyro sample site.
    It works much like TensorFlow's ``name_scope`` and ``variable_scope``,
    and can be used as a context manager, a decorator, or a higher-order function.

    ``scope`` is very useful for aligning compositional models with guides or data.

    Example::

        >>> @scope(prefix="a")
        ... def model():
        ...     return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "a/x" in poutine.trace(model).get_trace()


    Example::

        >>> def model():
        ...     with scope(prefix="a"):
        ...         return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "a/x" in poutine.trace(model).get_trace()

    Scopes compose as expected, with outer scopes appearing before inner scopes in names::

        >>> @scope(prefix="b")
        ... def model():
        ...     with scope(prefix="a"):
        ...         return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "b/a/x" in poutine.trace(model).get_trace()

    When used as a decorator or higher-order function,
    ``scope`` will use the name of the input function as the prefix
    if no user-specified prefix is provided.

    Example::

        >>> @scope
        ... def model():
        ...     return pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "model/x" in poutine.trace(model).get_trace()
    """
    msngr = ScopeMessenger(prefix=prefix, inner=inner)
    return msngr(fn) if fn is not None else msngr


def name_count(fn=None):
    """
    ``name_count`` is a very simple autonaming scheme that simply appends a suffix `"__"`
    plus a counter to any name that appears multiple tims in an execution.
    Only duplicate instances of a name get a suffix; the first instance is not modified.

    Example::

        >>> @name_count
        ... def model():
        ...     for i in range(3):
        ...         pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "x" in poutine.trace(model).get_trace()
        >>> assert "x__1" in poutine.trace(model).get_trace()
        >>> assert "x__2" in poutine.trace(model).get_trace()

    ``name_count`` also composes with :func:`~pyro.contrib.autoname.scope`
    by adding a suffix to duplicate scope entrances:

    Example::

        >>> @name_count
        ... def model():
        ...     for i in range(3):
        ...         with pyro.contrib.autoname.scope(prefix="a"):
        ...             pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "a/x" in poutine.trace(model).get_trace()
        >>> assert "a__1/x" in poutine.trace(model).get_trace()
        >>> assert "a__2/x" in poutine.trace(model).get_trace()

    Example::

        >>> @name_count
        ... def model():
        ...     with pyro.contrib.autoname.scope(prefix="a"):
        ...         for i in range(3):
        ...             pyro.sample("x", dist.Bernoulli(0.5))
        ...
        >>> assert "a/x" in poutine.trace(model).get_trace()
        >>> assert "a/x__1" in poutine.trace(model).get_trace()
        >>> assert "a/x__2" in poutine.trace(model).get_trace()
    """
    msngr = NameCountMessenger()
    return msngr(fn) if fn is not None else msngr
