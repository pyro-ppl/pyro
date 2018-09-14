from __future__ import absolute_import, division, print_function

import operator

from .messenger import Messenger
from .runtime import apply_stack, effectful


def _define_operators(c):
    for op in operator.__all__:
        if hasattr(operator, "__{}__".format(op)) and \
           callable(getattr(operator, op)) and \
           not hasattr(c, "__{}__".format(op)):
            # TODO pass a sensible name to effectful
            setattr(c, "__{}__".format(op),
                    effectful(getattr(operator, op)))
    return c


@_define_operators
class SimpleBox(object):
    """
    Wrapper without nesting weirdness
    """
    # TODO define some comparison methods?

    def __init__(self, value):
        assert not isinstance(value, SimpleBox), "dont need to wrap twice"
        self.value = value

    def __call__(self, *args, **kwargs):
        assert callable(self.value)
        # unbox and rebox
        return type(self)(value=effectful(self.value)(
            *(getattr(a, "value", a) for a in args),
            **{k: getattr(v, "value", v) for k, v in kwargs.items()}))


class SimpleNonstandardMessenger(Messenger):
    """
    Much-simplified version of NonstandardMessenger
    Does not do any weird nesting of value wrappers.
    """
    value_wrapper = SimpleBox

    def _process_message(self, msg):
        # TODO wrap all unwrapped things

        return super(SimpleNonstandardMessenger, self)._process_message(msg)
