from __future__ import absolute_import, division, print_function

import operator

from .messenger import Messenger
from .runtime import apply_stack


def make_nonstandard(fn):
    """
    Wrapper for calling apply_stack to apply any active effects.
    """
    def _fn(*args, **kwargs):
        msg = {
            "type": "apply",
            "name": kwargs.get("name", None),
            "fn": fn,
            "is_observed": False,
            "args": args,
            "kwargs": kwargs,
            "value": None,
            "scale": 1.0,
            "mask": None,
            "cond_indep_stack": (),
            "done": False,
            "stop": False,
            "continuation": None
        }
        # apply the stack and return its return value
        apply_stack(msg)
        return msg["value"]
    return _fn


class Box(object):
    """
    Boxed value.
    Each new NonstandardMessenger subclass should define its own Box subclass.
    """
    def __init__(self, value):
        """
        _ptr is the internal pointer to the wrapped data
        """
        self._ptr = value

    @property
    def value(self):
        """computes the value inside the box"""
        return self._ptr

    def update(self, new_value):
        self._ptr = new_value

    def __call__(self, *args, **kwargs):
        assert callable(self.value)
        # unbox and rebox
        return type(self)(value=self.value(
            *(getattr(a, "value", a) for a in args),
            **{k: getattr(v, "value", v) for k, v in kwargs.items()}))


for op in operator.__all__:
    if hasattr(operator, "__{}__".format(op)) and \
       callable(getattr(operator, op)):
        setattr(Box, "__{}__".format(op),
                make_nonstandard(getattr(operator, op)))


class NonstandardMessenger(Messenger):
    """
    Compositional nonstandard interpretation messenger.
    Useful for lazy evaluation, dependency tracking, conjugacy, etc.

    Wrapping a function (ordering option #1):
    1. Box all unboxed inputs <-- _process_message / __call__
    2. Unbox all boxed inputs <-- _process_message
    3. Unbox boxed function <-- _process_message
    4. Call unboxed function on unboxed inputs <-- what happens here??
    5. Box unboxed output <-- _postprocess_message
    6. Return boxed output

    Wrapping a function (ordering option #2):
    1. Keep boxing things up in _process_message
    2. In default_process, apply the maximally boxed function
    3. Unbox things on the way down in _postprocess_message

    How should function registration work?
    """
    def __init__(self):
        super(NonstandardMessenger, self).__init__()
        self._wrapper_cell = {}

    # these will usually need to be redefined for each subclass
    value_wrapper = Box
    function_wrapper = Box

    # def __call__(self, fn):
    #     def _wraps(*args, **kwargs):
    #         with self:
    #             return fn(
    #                 *tuple(self.value_wrapper(a) for a in args),
    #                 **{k: self.value_wrapper(v) for k, v in kwargs.items()}
    #             )
    #     _wraps.msngr = self
    #     return _wraps

    def _process_message(self, msg):
        """
        Unbox all boxed inputs
        """
        # boxing of any unboxed function and inputs
        if not isinstance(msg["fn"], self.function_wrapper):
            msg["fn"] = self.function_wrapper(msg["fn"])
        msg["args"] = tuple(a if isinstance(a, self.value_wrapper)
                            else self.value_wrapper(a)
                            for a in msg["args"])
        if msg["value"] is not None:
            msg["value"] = self.value_wrapper(msg["value"])

        # validation of boxed arguments
        assert isinstance(msg["fn"], self.function_wrapper)
        assert all(isinstance(x, self.value_wrapper) for x in msg["args"])
        assert all(isinstance(x, self.value_wrapper)
                   for x in msg["kwargs"].values())

        # validate type of value
        assert isinstance(msg["value"], self.value_wrapper) or not msg["value"]

        # store boxed values for postprocessing
        self._wrapper_cell["fn"] = msg["fn"]
        self._wrapper_cell["args"] = msg["args"]
        self._wrapper_cell["kwargs"] = msg["kwargs"]
        if msg["value"] is not None:
            self._wrapper_cell["value"] = msg["value"]

        # unbox values for function application
        msg["fn"] = msg["fn"]._ptr
        msg["args"] = tuple(arg._ptr for arg in msg["args"])
        msg["kwargs"] = {name: kwarg._ptr
                         for name, kwarg in msg["kwargs"].items()}
        if msg["value"] is not None:
            msg["value"] = msg["value"]._ptr

    def _postprocess_message(self, msg):
        """
        Re-boxing.
        """
        # validation of wrapped values
        assert all(self._wrapper_cell.get(field, None) is not None
                   for field in ["fn", "args", "kwargs"])

        # capture unboxed value
        new_val = msg["value"]

        # restore boxed values
        msg.update(self._wrapper_cell)
        self._wrapper_cell.clear()

        # application of boxed function
        # (this is where the actual effect is applied)
        msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])
        msg["value"].update(new_val)  # update value pointer

    def _reset(self):
        self._wrapper_cell.clear()
        return super(NonstandardMessenger, self)._reset()


class LazyBox(Box):

    def __init__(self, value=None, expr=()):
        self._ptr = value
        self._value = None
        self._expr = expr

    @property
    def value(self):
        if self._value is None:
            if self._expr:
                self._value = self._expr[0].value(
                    *(a.value for a in self._expr[1]),
                    **{k: v.value for k, v in self._expr[2].items()})
            else:
                self._value = self._ptr
        return self._value

    def __call__(self, *args, **kwargs):
        return LazyBox(expr=(self, args, kwargs))


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
    function_wrapper = LazyBox

    def _process_message(self, msg):
        msg["done"] = True  # block default processing
        super(LazyMessenger, self)._process_message(msg)


class ProvenanceBox(Box):

    @property
    def parents(self):
        return self._parents

    def __call__(self, *args, **kwargs):
        # XXX why is this failing with infinite recursion?
        # value = super(ProvenanceBox, self).__call__(*args, **kwargs)
        value = ProvenanceBox(None)
        value._parents = tuple(args) + tuple(kwargs.items())
        return value


class ProvenanceMessenger(NonstandardMessenger):
    """
    Provenance tracking - the absolute simplest nonstandard interpretation
    """
    # value wrappers
    value_wrapper = ProvenanceBox
    function_wrapper = ProvenanceBox
