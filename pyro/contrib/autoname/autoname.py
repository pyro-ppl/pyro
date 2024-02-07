# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterable
from functools import singledispatch

import pyro
from pyro.poutine.handlers import _make_handler
from pyro.poutine.reentrant_messenger import ReentrantMessenger
from pyro.poutine.runtime import effectful


@effectful(type="genname")
def genname(name="name"):
    return name


class NameScope:
    def __init__(self, name=None):
        self.name = name
        self.counter = 0
        self._namespace = defaultdict(int)

    def __str__(self):
        if self.counter:
            return f"{self.name}__{self.counter}"
        return str(self.name)

    def allocate(self, name):
        counter = self._namespace[name]
        self._namespace[name] += 1
        return counter


class ScopeStack:
    """
    Single global state to keep track of scope stacks.
    """

    def __init__(self):
        self._stack = []

    def __str__(self):
        return "/".join(str(scope) for scope in self._stack)

    @property
    def global_scope(self):
        return NameScope()  # don't keep a counter for a global scope

    @property
    def current_scope(self):
        if len(self._stack):
            return self._stack[-1]
        return self.global_scope

    def push_scope(self, scope):
        scope.counter = self.current_scope.allocate(scope.name)
        self._stack.append(scope)

    def pop_scope(self):
        return self._stack.pop(-1)

    def fresh_name(self, name):
        counter = self.current_scope.allocate(name)
        if counter:
            return name + str(counter)
        return name


class AutonameMessenger(ReentrantMessenger):
    """
    Assign unique names to random variables.

    1. For a new varialbe use its declared name if given, otherwise use the distribution name::

        sample("x", dist.Bernoulli ... )  # -> x
        sample(dist.Bernoulli ... )  # -> Bernoulli

    2. For repeated variables names append the counter as a suffix::

        sample(dist.Bernoulli ... )  # -> Bernoulli
        sample(dist.Bernoulli ... )  # -> Bernoulli1
        sample(dist.Bernoulli ... )  # -> Bernoulli2

    3. Functions and iterators can be used as a name scope::

        @autoname
        def f1():
            sample(dist.Bernoulli ... )

        @autoname
        def f2():
            f1()  # -> f2/f1/Bernoulli
            f1()  # -> f2/f1__1/Bernoulli
            sample(dist.Bernoulli ... )  # -> f2/Bernoulli

        @autoname(name="model")
        def f3():
            for i in autoname(range(3), name="time"):
                # model/time/Bernoulli .. model/time__1/Bernoulli .. model/time__2/Bernoulli
                sample(dist.Bernoulli ... )
                # model/time/f1/Bernoulli .. model/time__1/f1/Bernoulli .. model/time__2/f1/Bernoulli
                f1()

    4. Or scopes can be added using the with statement::

        def f4():
            with autoname(name="prefix"):
                f1()  # -> prefix/f1/Bernoulli
                f1()  # -> prefix/f1__1/Bernoulli
                sample(dist.Bernoulli ... )  # -> prefix/Bernoulli
    """

    def __init__(self, name=None):
        self.name = name
        super().__init__()

    def __call__(self, fn_or_iter):
        if isinstance(fn_or_iter, Iterable):
            if self.name is None:
                self.name = fn_or_iter.name  # name of a sequential pyro.plate
            self._iter = fn_or_iter
            return self
        if callable(fn_or_iter):
            if self.name is None:
                self.name = fn_or_iter.__name__
            return super().__call__(fn_or_iter)
        raise ValueError(f"{fn_or_iter} has to be an iterable or a callable.")

    def __enter__(self):
        scope = NameScope(self.name)
        _SCOPE_STACK.push_scope(scope)
        return super().__enter__()

    def __exit__(self, *args):
        _SCOPE_STACK.pop_scope()
        return super().__exit__(*args)

    def __iter__(self):
        for i in self._iter:
            scope = NameScope(self.name)
            _SCOPE_STACK.push_scope(scope)
            yield i
            scope = _SCOPE_STACK.pop_scope()

    @staticmethod  # only depends on the global _SCOPE_STACK state, not self
    def _pyro_genname(msg):
        raw_name = msg["fn"](*msg["args"])
        fresh_name = _SCOPE_STACK.fresh_name(raw_name)

        msg["value"] = str(_SCOPE_STACK) + "/" + fresh_name
        msg["stop"] = True


@_make_handler(AutonameMessenger, __name__)
def autoname(fn=None, name=None): ...


@singledispatch
def sample(*args):
    raise NotImplementedError


@sample.register(str)
def _sample_name(name, fn, *args, **kwargs):  # the current syntax of pyro.sample
    name = genname(name)
    return pyro.sample(name, fn, *args, **kwargs)


@sample.register(pyro.distributions.Distribution)
def _sample_dist(fn, *args, **kwargs):
    name = kwargs.pop("name", None)
    name = genname(type(fn).__name__ if name is None else name)
    return pyro.sample(name, fn, *args, **kwargs)


_SCOPE_STACK = ScopeStack()
