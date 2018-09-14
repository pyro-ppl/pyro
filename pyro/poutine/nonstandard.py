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


def _define_operators(c):
    """
    Decorator for generating operator methods
    """
    for op in operator.__all__:
        if hasattr(operator, "__{}__".format(op)) and \
           callable(getattr(operator, op)) and \
           not hasattr(c, "__{}__".format(op)):
            typename = "__{}__".format(op)
            # TODO pass a sensible name to effectful
            setattr(c, "__{}__".format(op),
                    effectful(unwrap_args(getattr(operator, op)), type=typename))
    return c


def _register_operators(default, post=False):
    def _decorator(msngr):
        c = msngr.value_wrapper
        for op in operator.__all__:
            typename = "__{}__".format(op)
            if hasattr(operator, typename) and \
               callable(getattr(operator, op)) and \
               hasattr(c, typename) and \
               not hasattr(msngr, "_pyro_" + ("_post_" if post else "") + typename):
                # TODO pass a sensible name to effectful
                msngr.register(fn=default, type=typename)
        return msngr
    return _decorator


@_define_operators
class Box(object):
    """
    Wrapper for defining nonstandard interpretations with poutine
    """
    def __init__(self, value):
        assert not isinstance(value, Box), "dont need to wrap twice"
        self.value = value

    def __call__(self, *args, **kwargs):
        assert callable(self.value)
        # unbox and rebox
        return type(self)(unwrap_args(effectful(self.value))(*args, **kwargs))


@_register_operators(lambda msg: msg)
class NonstandardMessenger(Messenger):
    """
    Much-simplified version of NonstandardMessenger
    Does not do any weird nesting of value wrappers.
    """
    value_wrapper = Box
