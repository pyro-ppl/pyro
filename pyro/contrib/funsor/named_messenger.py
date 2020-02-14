# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import ExitStack  # python 3

from pyro.poutine.reentrant_messenger import ReentrantMessenger


# name_to_dim : dict, dim_to_name : dict
StackFrame = namedtuple('StackFrame', ['name_to_dim', 'dim_to_name'])


class DimStack:

    def __init__(self):
        self.stack = [StackFrame(name_to_dim=OrderedDict(), dim_to_name=OrderedDict())]

    def push(self, frame):
        self.stack.append(frame)

    def pop(self):
        assert len(self.stack) > 1, "cannot pop the global frame"
        return self.stack.pop()

    def get(self, offset):
        assert offset <= 0
        return self.stack[len(self.stack) + offset - 1]

    @property
    def current_frame(self):
        return self.stack[-1]

    @property
    def global_frame(self):
        return self.stack[0]

    def free_globals(self, names):
        global_frame = self.stack[0]
        for name in frozenset(names).intersection(global_frame.name_to_dim):
            global_frame.dim_to_name.pop(global_frame.name_to_dim.pop(name))


_DIM_STACK = DimStack()  # only one global instance


class NamedMessenger(ReentrantMessenger):
    """
    Handler for converting to/from funsors consistent with Pyro's positional batch dimensions.

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    """
    def __init__(self, history=1, keep=False):
        assert history >= 0
        if keep:
            raise NotImplementedError("TODO")
        self.history = history
        self._iterable = None
        super().__init__()

    def generator(self, iterable):
        self._iterable = iterable
        return self

    def __iter__(self):
        assert self._iterable is not None
        # TODO handle parents (one-to-many)
        with ExitStack() as stack:
            for value in self._iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        _DIM_STACK.push(StackFrame(name_to_dim=OrderedDict(), dim_to_name=OrderedDict()))
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        _DIM_STACK.pop()
        return super().__exit__(*args, **kwargs)

    def _pyro_to_data(self, msg):
        if len(msg["args"]) < 2:
            msg["args"] = msg["args"] + (OrderedDict(),)
        funsor_value, name_to_dim = msg["args"]

        # handling non-fresh vars: just look at the deepest context they appear in?
        name_to_dim.update({name: _DIM_STACK.current_frame.name_to_dim[name]
                            for name in funsor_value.inputs
                            if name in _DIM_STACK.current_frame.name_to_dim
                            and name not in name_to_dim})

        # handling fresh vars: pass responsibility for allocation up the _DIM_STACK
        fresh = frozenset(name for name in funsor_value.inputs
                          if name not in name_to_dim
                          and name not in _DIM_STACK.current_frame.name_to_dim)

        # allocate fresh dimensions in the parent frame
        parent_frame = _DIM_STACK.get(-self.history)
        for name in fresh:

            # allocation
            if name not in parent_frame.name_to_dim:
                dim = -1
                while dim in parent_frame.dim_to_name:
                    dim -= 1
                parent_frame.name_to_dim[name] = dim

            dim = parent_frame.name_to_dim[name]
            parent_frame.dim_to_name[dim] = name
            name_to_dim[name] = dim

        # copy fresh variables down the stack
        for offset in range(-self.history, 1):
            frame = _DIM_STACK.get(offset)
            frame.name_to_dim.update({name: name_to_dim[name] for name in fresh})
            frame.dim_to_name.update({name_to_dim[name]: name for name in fresh})

    def _pyro_to_funsor(self, msg):
        if len(msg["args"]) < 3:
            msg["args"] = msg["args"] + (OrderedDict(),)
        raw_value, output, dim_to_name = msg["args"]

        event_dim = len(output.shape)
        batch_dim = len(raw_value.shape) - event_dim

        # since we don't allow fresh positional dims, only look at current frame
        dim_to_name.update({dim: _DIM_STACK.current_frame.dim_to_name[dim]
                            for dim in range(-batch_dim, 0)
                            if dim not in dim_to_name
                            and dim in _DIM_STACK.current_frame.dim_to_name})


def named(fn=None, history=1, keep=False):
    """
    Handler for converting to/from funsors consistent with Pyro's positional batch dimensions.

    This function is a piece of syntactic sugar that can be used in a variety of ways:
    - as a context manager
    - as a decorator for recursive functions
    - as an iterator for markov chains

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their share"
    """
    if fn is None:
        # Used as a decorator with bound args
        return NamedMessenger(history=history, keep=keep)
    if not callable(fn):
        # Used as a generator
        return NamedMessenger(history=history, keep=keep).generator(iterable=fn)
    # Used as a decorator with bound args
    return NamedMessenger(history=history, keep=keep)(fn)
