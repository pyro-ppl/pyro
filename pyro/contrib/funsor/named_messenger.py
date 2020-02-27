# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import ExitStack  # python 3
from enum import Enum

from pyro.poutine.reentrant_messenger import ReentrantMessenger


# name_to_dim : dict, dim_to_name : dict, parents : List, iter_parents : List
StackFrame = namedtuple('StackFrame', [
    'name_to_dim',
    'dim_to_name',
    'parents',
    'iter_parents',
    'keep',
])


class DimType(Enum):
    """Enumerates the possible types of dimensions to allocate"""
    LOCAL = 0
    GLOBAL = 1
    VISIBLE = 2


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
        self._first_available_dim = -1

    MAX_DIM = -25

    def set_first_available_dim(self, dim):
        assert dim is None or (self.MAX_DIM < dim < 0)
        old_dim, self._first_available_dim = self._first_available_dim, dim
        return old_dim

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

    def _gendim(self, name, fresh_dim_to_name, dim_type=DimType.LOCAL):
        """generate a unique fresh positional dimension using the allocator state"""
        dim = self._first_available_dim if dim_type != DimType.VISIBLE else -1
        dim = -1 if dim is None else dim
        frames = [self.current_frame, self.global_frame] + \
            self.current_frame.parents + self.current_frame.iter_parents
        while dim in fresh_dim_to_name or any(dim in p.dim_to_name for p in frames):
            dim -= 1
        if dim < self.MAX_DIM or (dim_type == DimType.VISIBLE and dim <= self._first_available_dim):
            raise ValueError(f"Ran out of free dims during allocation for {name}")
        return dim

    def _gensym(self, dim, fresh_name_to_dim, dim_type=DimType.LOCAL):
        """deterministically generate a name following funsor.pyro.convert convention"""
        name = f"_pyro_dim_{-dim}"  # XXX -dim-1? dim? check this
        assert dim < 0 and name not in fresh_name_to_dim
        return name

    def allocate_dims(self, fresh_names, dim_type=DimType.LOCAL):

        fresh_name_to_dim = OrderedDict()
        fresh_dim_to_name = OrderedDict()

        for name in fresh_names:

            # allocation
            dim = None
            for parent_frame in self.current_frame.parents + [self.global_frame]:
                dim = parent_frame.name_to_dim.get(name, dim)

            if dim is None:
                dim = self._gendim(name, fresh_dim_to_name, dim_type=dim_type)

            fresh_dim_to_name[dim] = name
            fresh_name_to_dim[name] = dim

        # copy fresh variables down the stack
        frames = [self.global_frame] if dim_type != DimType.LOCAL else [self.current_frame] + \
            (self.current_frame.parents if self.current_frame.keep else [])
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
        msg.setdefault("dim_type", msg["kwargs"].pop("dim_type", DimType.LOCAL))
        funsor_value, name_to_dim = msg["args"]

        # handling non-fresh vars: just look at the deepest context they appear in
        for frame in [_DIM_STACK.current_frame] + \
                _DIM_STACK.current_frame.parents + _DIM_STACK.current_frame.iter_parents:
            name_to_dim.update({name: frame.name_to_dim[name]
                                for name in funsor_value.inputs
                                if name in frame.name_to_dim
                                and name not in name_to_dim})

        # handling fresh vars: pass responsibility for allocation up the _DIM_STACK
        fresh_names = tuple(name for name in funsor_value.inputs
                            if name not in name_to_dim
                            and name not in _DIM_STACK.current_frame.name_to_dim)

        # allocate fresh dimensions in the parent frame
        if fresh_names:
            name_to_dim.update(_DIM_STACK.allocate_dims(fresh_names, dim_type=msg["dim_type"]))

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_funsor(msg):

        if len(msg["args"]) < 3:
            msg["args"] = msg["args"] + (OrderedDict(),)
        msg.setdefault("dim_type", msg["kwargs"].pop("dim_type", DimType.LOCAL))
        raw_value, output, dim_to_name = msg["args"]

        event_dim = len(output.shape)
        batch_dim = len(raw_value.shape) - event_dim

        # handling non-fresh dims: just look at the deepest context they appear in
        # TODO support fresh dims
        for frame in [_DIM_STACK.current_frame] + \
                _DIM_STACK.current_frame.parents + _DIM_STACK.current_frame.iter_parents:
            dim_to_name.update({dim: frame.dim_to_name[dim]
                                for dim in range(-batch_dim, 0)
                                if raw_value.shape[dim - event_dim] > 1
                                and dim in frame.dim_to_name
                                and dim not in dim_to_name})


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

        frame.iter_parents[:] = self._iter_parents[:] + [_DIM_STACK.global_frame]
        if self.history > 0:
            frame.parents[:] = reversed(_DIM_STACK._stack[len(_DIM_STACK._stack) - self.history:])

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
        self.first_available_dim = first_available_dim
        super().__init__()

    def __enter__(self):
        if self._ref_count == 0:
            self._prev_first_dim = _DIM_STACK.set_first_available_dim(self.first_available_dim)
            self._extant_globals = frozenset(_DIM_STACK.global_frame.name_to_dim)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1:
            _DIM_STACK.set_first_available_dim(self._prev_first_dim)
            _DIM_STACK.free_globals(frozenset(_DIM_STACK.global_frame.name_to_dim) - self._extant_globals)
        return super().__exit__(*args, **kwargs)
