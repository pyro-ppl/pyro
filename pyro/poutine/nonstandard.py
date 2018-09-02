from __future__ import absolute_import, division, print_function

import pyro  # XXX circular dependency, replace sample with apply_stack
from .messenger import Messenger
from .runtime import apply_stack


def make_nonstandard(fn):
    """
    Wrapper for calling apply_stack to apply any active effects.
    """
    def _fn(*args, **kwargs):
        return pyro.sample("aa", fn, *args, **kwargs)  # XXX not really a sample
    return _fn


class Box(object):
    """
    Boxed value.
    Each new NonstandardMessenger subclass should define its own Box subclass.
    """
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


class BoxedCallable(Box):
    """
    Boxed function (which is also a boxed value)
    Each new NonstandardMessenger subclass should define
    its own BoxedCallable subclass.

    Should composition run in the other direction?
    ie should we just box Callables?
    """
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class NonstandardMessenger(Messenger):
    """
    Compositional nonstandard interpretation messenger.
    Useful for lazy evaluation, dependency tracking, conjugacy, etc.

    Wrapping a function:
    1. Box all unboxed inputs <-- _process_message / __call__
    2. Unbox all boxed inputs <-- _process_message
    3. Unbox boxed function <-- _process_message
    4. Call unboxed function on unboxed inputs <-- apply_stack (?)
    5. Box unboxed output <-- _postprocess_message
    6. Return boxed output
    """
    def __init__(self):
        super(NonstandardMessenger, self).__init__()
        self._wrapper_cell = {}

    # these usually need to be defined for each subclass
    value_wrapper = Box  # XXX should be a @property?
    function_wrapper = BoxedCallable  # XXX should be a @property?

    def _process_message(self, msg):
        """
        Unbox all boxed inputs
        """
        # boxing
        msg["fn"] = self.function_wrapper(msg["fn"])
        msg["args"] = tuple(a if isinstance(a, self.value_wrapper)
                            else self.value_wrapper(a)
                            for a in msg["args"])

        # validation
        assert isinstance(msg["fn"], self.function_wrapper)
        assert all(isinstance(x, self.value_wrapper) for x in msg["args"])
        assert all(isinstance(x, self.value_wrapper)
                   for x in msg["kwargs"].values())
        assert isinstance(msg["value"], self.value_wrapper) or not msg["value"]

        # store boxed values for postprocessing
        # XXX what about msg["value"]?
        self._wrapper_cell["fn"] = msg["fn"]
        self._wrapper_cell["args"] = msg["args"]
        self._wrapper_cell["kwargs"] = msg["kwargs"]
        if msg["value"] is not None:
            self._wrapper_cell["value"] = msg["value"]

        # unbox values for function application
        # XXX what about msg["value"]?
        msg["fn"] = msg["fn"].value
        msg["args"] = tuple(arg.value for arg in msg["args"])
        msg["kwargs"] = {name: kwarg.value
                         for name, kwarg in msg["kwargs"].items()}
        if msg["value"] is not None:
            msg["value"] = msg["value"].value

    def _postprocess_message(self, msg):
        """
        Re-boxing.
        """
        # validation
        assert all(self._wrapper_cell.get(field, None) is not None
                   for field in ["fn", "args", "kwargs"])

        # restore boxed values
        msg.update(self._wrapper_cell)

    def _reset(self):
        self._wrapper_cell.clear()
        return super(NonstandardMessenger, self)._reset()


class LazyBox(Box):

    @property
    def value(self):
        pass


class LazyBoxedCallable(BoxedCallable):

    def __call__(self, *args, **kwargs):
        pass


class LazyMessenger(NonstandardMessenger):
    """
    A NonstandardMessenger version of lazy evaluation.
    Question: can this be applied below any other NonstandardMessengers?
    Ok, so laziness is an especially tricky one...?
    We should be able to abuse the "done" control flag here

    Mechanism: attach a tape in the BoxedCallable subclass
    """

    # value wrappers for lazy evaluation
    value_wrapper = LazyBox
    function_wrapper = LazyBoxedCallable


class ProvenanceBox(Box):

    def set_parents(self, *args, **kwargs):
        self._parents = tuple(args) + tuple(kwargs.items())


class ProvenanceBoxedCallable(BoxedCallable):

    def __call__(self, *args, **kwargs):
        value = ProvenanceBox(self.fn(*args, **kwargs))
        value.set_parents(tuple(args) + tuple(kwargs.items()))
        return value


class ProvenanceMessenger(NonstandardMessenger):
    """
    Provenance tracking - the absolute simplest nonstandard interpretation
    """
    # value wrappers
    value_wrapper = ProvenanceBox
    function_wrapper = ProvenanceBoxedCallable
