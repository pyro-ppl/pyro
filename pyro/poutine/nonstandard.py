from __future__ import absolute_import, division, print_function

import operator

from .messenger import Messenger
from .runtime import effectful


def unwrap_args(fn):
    def _fn(*args, **kwargs):
        args = tuple(a.value if isinstance(a, Box) else a for a in args)
        kwargs = {k: v.value if isinstance(v, Box) else v for k, v in kwargs.items()}
        return fn(*args, **kwargs)
    return _fn


def rewrap_ret(cls, fn=None):
    if fn is None:
        return lambda x: rewrap_ret(cls, x)

    def _fn(*args, **kwargs):
        return cls(fn(*args, **kwargs))
    return _fn


def _define_operators(c):
    """
    Decorator for generating operator methods
    """
    for op in operator.__all__:
        if hasattr(operator, "__{}__".format(op)) and \
           callable(getattr(operator, op)) and \
           not hasattr(c, "__{}__".format(op)):
            typename = "__{}__".format(op)
            setattr(c, "__{}__".format(op),
                    rewrap_ret(c, effectful(unwrap_args(getattr(operator, op)),
                                            type=typename)))
    return c


def _register_operators(default, post=False, overwrite=True):
    def _decorator(msngr):
        for op in operator.__all__:
            typename = "__{}__".format(op)
            if hasattr(operator, typename) and \
               callable(getattr(operator, op)):
                if overwrite or not \
                   hasattr(msngr, "_pyro_" + ("_post_" if post else "") + typename):
                    msngr.register(fn=default, type=typename)
        return msngr
    return _decorator


@_define_operators
class Box(object):
    """
    Wrapper for defining nonstandard interpretations with poutine
    """
    def __init__(self, value, typename=None):
        assert not isinstance(value, Box), "dont need to wrap twice"
        self.value = value
        self.typename = typename

    def __getattribute__(self, key):
        # cases
        if key == "value" or \
           (key in operator.__all__ and callable(getattr(operator, key))):
            return super(Box, self).__getattribute__(key)
        elif hasattr(self.value, key):
            val = getattr(self.value, key)
            if callable(val):
                return rewrap_ret(
                    type(self),
                    effectful(unwrap_args(val), type=key))
            return type(self)(val)
        return super(Box, self).__getattribute__(key)

    def __call__(self, *args, **kwargs):
        assert callable(self.value)
        # unbox and rebox
        return rewrap_ret(
            type(self), effectful(unwrap_args(self.value), type=self.typename)
        )(*args, **kwargs)


@_register_operators(lambda msg: msg)
class NonstandardMessenger(Messenger):
    """
    Much-simplified version of NonstandardMessenger
    Does not do any weird nesting of value wrappers.
    """
    pass


class LazyValue(object):
    def __init__(self, fn, *args, **kwargs):
        if getattr(fn, "__wrapped", False):
            raise NotImplementedError
        self._expr = (fn, args, kwargs)
        self._value = None

    def eval(self):
        if self._value is None:
            fn = self._expr[0]
            args = tuple(a.value.eval() if isinstance(a.value, LazyValue) else a
                         for a in self._expr[1])
            kwargs = {k: v.value.eval() if isinstance(v.value, LazyValue) else v
                      for k, v in self._expr[2].items()}
            self._value = fn(*args, **kwargs)
            self._expr = None
        return self._value


def lazy_wrap(msg):
    msg["value"] = LazyValue(msg["fn"], *msg["args"], **msg["kwargs"])
    msg["done"] = True
    return msg


@_register_operators(lazy_wrap)
class LazyMessenger(NonstandardMessenger):
    """
    Messenger for lazy evaluation
    """
    def _process_message(self, msg):
        super(LazyMessenger, self)._process_message(msg)
        if not msg["done"] and msg["fn"] is not None:
            return lazy_wrap(msg)
        return msg
