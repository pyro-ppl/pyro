import collections
import copy
import functools

from pyro.poutine.reentrant_messenger import ReentrantMessenger
from pyro.poutine.runtime import effectful
import pyro


@effectful(type="genname")
def genname(name="name"):
    return name


_AUTONAME_STACK = []
_AUTONAME_LOCAL_COUNTERS = collections.defaultdict(int)
_AUTONAME_CALL_COUNTERS = collections.defaultdict(int)


class AutonameFrame:
    def __init__(self, name, counter):
        self.name = name
        self.counter = counter

    def __hash__(self):
        return hash(self.name)
        # return hash((self.name, self.counter))

    def __eq__(self, other):
        return type(self) == type(other) and (self.name, self.counter) == (other.name, other.counter)


class AutonameMessenger(ReentrantMessenger):
    """
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
        self.frame = AutonameFrame(name, 0)
        super().__init__()

    def __call__(self, fn_or_iter):
        if callable(fn_or_iter):
            if self.frame.name is None:
                self.frame.name = fn_or_iter.__name__
            return super().__call__(effectful(type="call_scope")(fn_or_iter))
        self._iter = fn_or_iter
        return self

    def _pyro_genname(self, msg):
        # example: genname() -> model_0/time_0/var0, model_0/time_0/var1, model_0/time_1/var0
        raw_name = msg["fn"](*msg["args"])
        
        context = tuple(_AUTONAME_STACK)
        breakpoint()
        msg["value"] = "/".join("{}_{}".format(frame.name, frame.counter) for frame in _AUTONAME_STACK) + \
            "/" + raw_name + "_" + str(_AUTONAME_LOCAL_COUNTERS[context])
        _AUTONAME_LOCAL_COUNTERS[context] += 1
        msg["stop"] = True
        msg["done"] = True

    def _pyro_call_scope(self, msg):
        # breakpoint()
        _AUTONAME_STACK.append(copy.copy(self.frame))
        context = tuple(hash(frame) for frame in _AUTONAME_STACK)
        _AUTONAME_CALL_COUNTERS[context] += 1
        self.frame.counter = _AUTONAME_CALL_COUNTERS[context]
        msg["stop"] = True

    def _pyro_post_call_scope(self, msg):
        # breakpoint()
        # deleted_context = tuple(_AUTONAME_STACK)
        deleted_context = tuple(hash(frame) for frame in _AUTONAME_STACK)
        #  _AUTONAME_CALL_COUNTERS[deleted_context] -= 1
        #  self.frame.counter = _AUTONAME_CALL_COUNTERS[deleted_context]
        #  _AUTONAME_CALL_COUNTERS.pop(deleted_context, None)
        #  _AUTONAME_LOCAL_COUNTERS.pop(deleted_context, None)
        _AUTONAME_STACK.pop(-1)
        msg["stop"] = True

    def __iter__(self):
        with self:
            _AUTONAME_STACK.append(self.frame)
            for i in self._iter:
                self.frame.counter = i
                yield i
            deleted_context = tuple(_AUTONAME_STACK)
            _AUTONAME_CALL_COUNTERS.pop(deleted_context, None)
            _AUTONAME_LOCAL_COUNTERS.pop(deleted_context, None)
            _AUTONAME_STACK.pop(-1)
            self.frame.counter = 0


def autoname(fn=None, name=None):
    msngr = AutonameMessenger(name=name)
    return msngr(fn) if fn is not None else msngr


@functools.singledispatch
def sample(*args):
    raise NotImplementedError


@sample.register(str)
def _sample_name(name, d, *args, **kwargs):  # the current syntax of pyro.sample
    return pyro.sample(name, d, *args, **kwargs)


@sample.register(pyro.distributions.Distribution)
def _sample_dist(d, *args, **kwargs):
    name = kwargs.pop("name", None)
    name = genname(type(d).__name__ if name is None else name)
    return pyro.sample(name, d, *args, **kwargs)
