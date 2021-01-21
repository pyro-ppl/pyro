import functools
from collections import defaultdict

import pyro
from pyro.poutine.reentrant_messenger import ReentrantMessenger
from pyro.poutine.runtime import effectful


@effectful(type="genname")
def genname(name="name"):
    return name


class NameScope:
    def __init__(self, name=None):
        self.name = name
        self.counter = 0
        self._inner_scopes = defaultdict(int)

    @property
    def full_name(self):
        if self.counter:
            return f"{self.name}__{self.counter}"
        return self.name


class ScopeStack:
    """
    Single global state to keep track of scope stacks.
    """

    def __init__(self):
        self._stack = []

    @property
    def local_scope(self):
        if len(self._stack):
            return self._stack[-1]
        return NameScope()

    def push_scope(self, scope):
        scope.counter = self.local_scope._inner_scopes[scope.name]
        self.local_scope._inner_scopes[scope.name] += 1
        self._stack.append(scope)

    def pop_scope(self):
        return self._stack.pop(-1)

    def new_name(self, name):
        counter = self.local_scope._inner_scopes[name]
        self.local_scope._inner_scopes[name] += 1
        if counter:
            return name + str(counter)
        return name

    @property
    def full_scope_name(self):
        return "/".join(scope.full_name for scope in self._stack)


class AutonameMessenger(ReentrantMessenger):
    """
    Naming random variables.

    @autoname
    def model(x):
        for i in autoname(vectorized_markov(range(3), name="time")):
            print(genname())  # -> ...var1...
            print(genname())  # -> ...var2...
            sample(genname(), ...)
            param(genname(), ...)


    @autoname
    def f1():
        f3()

    @autoname
    def f3():
        ...

    @autoname
    def f2():
        f1()  # f2_0/f1_0, f2_0/f1_0/f3_0
        f1()  # f2_0/f1_1, f2_0/f1_1/f3_1 (should be f2_0/f1_1/f3_0)
        f3()  # currently f2_0/f3_3 (should be f2_0/f3_0)
    """

    def __init__(self, name=None):
        self.scope = NameScope(name)
        super().__init__()

    def __call__(self, fn_or_iter):
        # reuse sequential pyro.plate name
        # also get suffix names from pyro.vectorized_plate and add to NameScope?
        if hasattr(fn_or_iter, "name"):
            self.scope.name = fn_or_iter.name
            self._iter = fn_or_iter
            return self
        if callable(fn_or_iter):
            if self.scope.name is None:
                self.scope.name = fn_or_iter.__name__
            return super().__call__(effectful(type="call_scope")(fn_or_iter))
        self._iter = fn_or_iter
        return self

    @staticmethod  # only depends on the global _SCOPE_STACK state, not self
    def _pyro_genname(msg):
        # example: genname() -> model_0/time_0/var0, model_0/time_0/var1, model_0/time_1/var0
        raw_name = msg["fn"](*msg["args"])
        new_name = _SCOPE_STACK.new_name(raw_name)

        msg["value"] = _SCOPE_STACK.full_scope_name + "/" + new_name
        msg["stop"] = True

    def _pyro_call_scope(self, msg):
        _SCOPE_STACK.push_scope(self.scope)
        msg["stop"] = True

    def _pyro_post_call_scope(self, msg):
        scope = _SCOPE_STACK.pop_scope()
        scope._inner_scopes = defaultdict(int)
        msg["stop"] = True

    def __iter__(self):
        with self:
            for i in self._iter:
                _SCOPE_STACK.push_scope(self.scope)
                yield i
                scope = _SCOPE_STACK.pop_scope()
                scope._inner_scopes = defaultdict(int)


def autoname(fn=None, name=None):
    msngr = AutonameMessenger(name=name)
    return msngr(fn) if fn is not None else msngr


@functools.singledispatch
def sample(*args):
    raise NotImplementedError


@sample.register(str)
def _sample_name(name, d, *args, **kwargs):  # the current syntax of pyro.sample
    name = genname(name)
    return pyro.sample(name, d, *args, **kwargs)


@sample.register(pyro.distributions.Distribution)
def _sample_dist(d, *args, **kwargs):
    name = kwargs.pop("name", None)
    name = genname(type(d).__name__ if name is None else name)
    return pyro.sample(name, d, *args, **kwargs)


_SCOPE_STACK = ScopeStack()
