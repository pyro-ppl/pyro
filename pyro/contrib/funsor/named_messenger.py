# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import ExitStack  # python 3

from pyro.poutine.reentrant_messenger import ReentrantMessenger


# name_to_dim : dict, dim_to_name : dict, parents : List, iter_parents : List
StackFrame = namedtuple('StackFrame', [
    'name_to_dim',
    'dim_to_name',
    'parents',
    'iter_parents',
    'keep',
])


class DimStack:
    """
    Single piece of global state to keep track of the mapping between names and dimensions.

    Replaces the plate DimAllocator, the enum EnumAllocator, the stack in MarkovMessenger,
    _param_dims and _value_dims in EnumMessenger, and dim_to_symbol in msg['infer']
    """
    def __init__(self):
        self._stack = [StackFrame(
            name_to_dim=OrderedDict(), dim_to_name=OrderedDict(),
            parents=[], iter_parents=[], keep=False,
        )]
        self._first_available_dim = None  # TODO default to -1?

    def set_first_available_dim(self, dim):
        self._first_available_dim = dim

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

    def allocate_dims(self, fresh, visible=False):

        if visible:
            raise NotImplementedError("TODO implement plate dimension allocation")

        fresh_name_to_dim = OrderedDict()
        fresh_dim_to_name = OrderedDict()

        for name in fresh:

            # allocation
            dim = None
            for parent_frame in self.current_frame.parents:  # don't read iter_parents
                dim = parent_frame.name_to_dim.get(name, None)

            if dim is None:  # find a unique dimension
                dim = -1
                while dim in fresh_dim_to_name or \
                        dim in self.current_frame.dim_to_name or \
                        any(dim in p.dim_to_name for p in self.current_frame.parents) or \
                        any(dim in p.dim_to_name for p in self.current_frame.iter_parents):
                    dim -= 1

            fresh_dim_to_name[dim] = name
            fresh_name_to_dim[name] = dim

        # copy fresh variables down the stack
        frames = [self.current_frame]
        frames += self.current_frame.parents if self.current_frame.keep else []
        for frame in frames:
            frame.name_to_dim.update(fresh_name_to_dim)
            frame.dim_to_name.update(fresh_dim_to_name)

        return fresh_name_to_dim


_DIM_STACK = DimStack()  # only one global instance


class NamedMessenger(ReentrantMessenger):

    @staticmethod
    def _pyro_to_data(msg):

        if len(msg["args"]) < 2:
            msg["args"] = msg["args"] + (OrderedDict(),)
        funsor_value, name_to_dim = msg["args"]

        # handling non-fresh vars: just look at the deepest context they appear in
        for frame in [_DIM_STACK.current_frame] + _DIM_STACK.current_frame.parents:
            name_to_dim.update({name: frame.name_to_dim[name]
                                for name in funsor_value.inputs
                                if name in frame.name_to_dim
                                and name not in name_to_dim})

        # handling fresh vars: pass responsibility for allocation up the _DIM_STACK
        fresh = frozenset(name for name in funsor_value.inputs
                          if name not in name_to_dim
                          and name not in _DIM_STACK.current_frame.name_to_dim)

        # allocate fresh dimensions in the parent frame
        if fresh:
            name_to_dim.update(_DIM_STACK.allocate_dims(fresh))

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_funsor(msg):

        if len(msg["args"]) < 3:
            msg["args"] = msg["args"] + (OrderedDict(),)
        raw_value, output, dim_to_name = msg["args"]

        event_dim = len(output.shape)
        batch_dim = len(raw_value.shape) - event_dim

        # handling non-fresh dims: just look at the deepest context they appear in
        # TODO support fresh dims
        for frame in [_DIM_STACK.current_frame] + _DIM_STACK.current_frame.parents:
            dim_to_name.update({dim: frame.dim_to_name[dim]
                                for dim in range(-batch_dim, 0)
                                if dim not in dim_to_name
                                and dim in frame.dim_to_name})


class LocalNamedMessenger(NamedMessenger):
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
        self.history = history
        self.keep = keep
        self._iterable = None
        self._saved_frames = []
        self._iter_parents = []
        super().__init__()

    def generator(self, iterable):
        self._iterable = iterable
        return self

    def _get_parents(self, frame):
        # TODO get this right to match Pyro allocation order
        return reversed(_DIM_STACK._stack[len(_DIM_STACK._stack) - self.history:])

    def _get_iter_parents(self, frame):
        iter_parents = [frame]
        frontier = [frame]
        while frontier:
            frontier = sum([p.iter_parents for p in frontier], [])
            iter_parents += frontier
        return iter_parents

    def __iter__(self):
        assert self._iterable is not None
        self._iter_parents = self._get_iter_parents(_DIM_STACK.current_frame)
        with ExitStack() as stack:
            for value in self._iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        if self.keep and self._saved_frames:
            frame = self._saved_frames.pop()
        else:
            frame = StackFrame(name_to_dim=OrderedDict(), dim_to_name=OrderedDict(),
                               parents=[], iter_parents=[], keep=self.keep)

        frame.iter_parents[:] = self._iter_parents[:]
        if self.history > 0:
            frame.parents[:] = self._get_parents(_DIM_STACK.current_frame)

        _DIM_STACK.push(frame)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self.keep:
            # don't keep around references to other frames
            saved_frame = _DIM_STACK.pop()
            saved_frame.parents[:] = []
            saved_frame.iter_parents[:] = []
            self._saved_frames.append(saved_frame)
        else:
            _DIM_STACK.pop()
        return super().__exit__(*args, **kwargs)


class GlobalNamedMessenger(NamedMessenger):

    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        if first_available_dim is not None and first_available_dim < -1:
            raise NotImplementedError("TODO support plates")  # TODO
        self.first_available_dim = first_available_dim  # TODO store this in _DIM_STACK
        super().__init__()

    def __enter__(self):
        if self._ref_count == 0:
            # TODO set _DIM_STACK._first_available_dim
            self._extant_globals = frozenset(_DIM_STACK.global_frame.name_to_dim)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1:
            # TODO unset _DIM_STACK._first_available_dim
            _DIM_STACK.free_globals(frozenset(_DIM_STACK.global_frame.name_to_dim) - self._extant_globals)
        return super().__exit__(*args, **kwargs)
