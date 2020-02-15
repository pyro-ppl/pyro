# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import ExitStack  # python 3

from pyro.poutine.messenger import Messenger
from pyro.poutine.reentrant_messenger import ReentrantMessenger


# name_to_dim : dict, dim_to_name : dict
StackFrame = namedtuple('StackFrame', ['name_to_dim', 'dim_to_name'])


class DimStack:

    def __init__(self):
        self._stack = [StackFrame(name_to_dim=OrderedDict(), dim_to_name=OrderedDict())]

    def push(self, frame):
        self._stack.append(frame)

    def pop(self):
        assert len(self._stack) > 1, "cannot pop the global frame"
        return self._stack.pop()

    @property
    def current_frame(self):
        return self._stack[-1]

    @property
    def global_frame(self):
        return self._stack[0]

    def free_globals(self, names=None):
        global_frame = self.global_frame
        names = frozenset(global_frame.name_to_dim.keys()) if names is None else names
        for name in frozenset(names).intersection(global_frame.name_to_dim):
            global_frame.dim_to_name.pop(global_frame.name_to_dim.pop(name))

    def allocate(self, fresh, history=1, visible=False):
        if visible:
            raise NotImplementedError("TODO implement plate dimension allocation")

        fresh_name_to_dim = OrderedDict()

        parent_frame = self._stack[len(self._stack) - history - 1]
        for name in fresh:

            # allocation
            if name not in parent_frame.name_to_dim:
                dim = -1
                while dim in parent_frame.dim_to_name:
                    dim -= 1
                parent_frame.name_to_dim[name] = dim

            dim = parent_frame.name_to_dim[name]
            parent_frame.dim_to_name[dim] = name
            fresh_name_to_dim[name] = dim

        # copy fresh variables down the stack
        for offset in range(-history, 1):
            frame = self._stack[len(self._stack) + offset - 1]
            frame.name_to_dim.update({name: fresh_name_to_dim[name] for name in fresh})
            frame.dim_to_name.update({fresh_name_to_dim[name]: name for name in fresh})

        return fresh_name_to_dim


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
        self.history = history
        self.keep = keep
        self._iterable = None
        self._saved_frames = []
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
        if self.keep and self._saved_frames:
            _DIM_STACK.push(self._saved_frames.pop())
        else:
            _DIM_STACK.push(StackFrame(name_to_dim=OrderedDict(), dim_to_name=OrderedDict()))
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self.keep:
            self._saved_frames.append(_DIM_STACK.pop())
        else:
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
        name_to_dim.update(_DIM_STACK.allocate(fresh, self.history))

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


class GlobalNamedMessenger(Messenger):
    # demonstration of managing names in global scope, as done by plate and enum
    def __enter__(self):
        self._extant_globals = frozenset(_DIM_STACK.global_frame.name_to_dim)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        _DIM_STACK.free_globals(frozenset(_DIM_STACK.global_frame.name_to_dim) - self._extant_globals)
        return super().__exit__(*args, **kwargs)
