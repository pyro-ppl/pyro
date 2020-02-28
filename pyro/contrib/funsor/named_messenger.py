# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import ExitStack  # python 3
from enum import Enum

from pyro.poutine.reentrant_messenger import ReentrantMessenger


# name_to_dim : dict, dim_to_name : dict, parents : tuple, iter_parents : tuple
class StackFrame(namedtuple('StackFrame', [
            'name_to_dim',
            'dim_to_name',
            'parents',
            'iter_parents',
            'keep',
        ])):

    def read(self, name, dim):
        found_name = self.dim_to_name.get(dim, name)
        found_dim = self.name_to_dim.get(name, dim)
        found = name in self.name_to_dim or dim in self.dim_to_name
        return found_name, found_dim, found

    def write(self, name, dim):
        assert name is not None and dim is not None
        self.dim_to_name[dim] = name
        self.name_to_dim[name] = dim

    def free(self, name, dim):
        self.dim_to_name.pop(dim, None)
        self.name_to_dim.pop(name, None)
        return name, dim


class DimType(Enum):
    """Enumerates the possible types of dimensions to allocate"""
    LOCAL = 0
    GLOBAL = 1
    VISIBLE = 2


DimRequest = namedtuple('DimRequest', ['dim', 'dim_type'], defaults=(None, DimType.LOCAL))
NameRequest = namedtuple('NameRequest', ['name', 'dim_type'], defaults=(None, DimType.LOCAL))


class DimStack:
    """
    Single piece of global state to keep track of the mapping between names and dimensions.

    Replaces the plate DimAllocator, the enum EnumAllocator, the stack in MarkovMessenger,
    _param_dims and _value_dims in EnumMessenger, and dim_to_symbol in msg['infer']
    """
    def __init__(self):
        self._stack = [StackFrame(
            name_to_dim=OrderedDict(), dim_to_name=OrderedDict(),
            parents=(), iter_parents=(), keep=False,
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

    def _gendim(self, name_request, dim_request):
        assert isinstance(name_request, NameRequest) and isinstance(dim_request, DimRequest)
        dim_type = dim_request.dim_type

        if name_request.name is None:
            fresh_name = f"_pyro_dim_{-dim_request.dim}"
        else:
            fresh_name = name_request.name

        conflict_frames = (self.current_frame, self.global_frame) + \
            self.current_frame.parents + self.current_frame.iter_parents
        if dim_request.dim is None:
            fresh_dim = self._first_available_dim if dim_type != DimType.VISIBLE else -1
            fresh_dim = -1 if fresh_dim is None else fresh_dim
            while any(fresh_dim in p.dim_to_name for p in conflict_frames):
                fresh_dim -= 1
        else:
            fresh_dim = dim_request.dim

        if fresh_dim < self.MAX_DIM or \
                any(fresh_dim in p.dim_to_name for p in conflict_frames) or \
                (dim_type == DimType.VISIBLE and fresh_dim <= self._first_available_dim):
            raise ValueError(f"Ran out of free dims during allocation for {fresh_name}")

        return fresh_name, fresh_dim

    def request(self, name, dim):
        assert isinstance(name, NameRequest) ^ isinstance(dim, DimRequest)
        if isinstance(dim, DimRequest):
            dim, dim_type = dim.dim, dim.dim_type
        elif isinstance(name, NameRequest):
            name, dim_type = name.name, name.dim_type

        read_frames = (self.global_frame,) if dim_type != DimType.LOCAL else \
            (self.current_frame,) + self.current_frame.parents + self.current_frame.iter_parents + (self.global_frame,)

        # read dimension
        for frame in read_frames:
            name, dim, found = frame.read(name, dim)
            if found:
                break

        # generate fresh name or dimension
        if not found:
            name, dim = self._gendim(NameRequest(name, dim_type), DimRequest(dim, dim_type))

            write_frames = (self.global_frame,) if dim_type != DimType.LOCAL else \
                (self.current_frame,) + (self.current_frame.parents if self.current_frame.keep else ())

            # store the fresh dimension
            for frame in write_frames:
                frame.write(name, dim)

        return name, dim


_DIM_STACK = DimStack()  # only one global instance


class NamedMessenger(ReentrantMessenger):

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_data(msg):

        funsor_value, = msg["args"]
        name_to_dim = msg["kwargs"].setdefault("name_to_dim", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        # interpret all names/dims as requests since we only run this function once
        for name in funsor_value.inputs:
            dim = name_to_dim.get(name, None)
            name_to_dim[name] = dim if isinstance(dim, DimRequest) else DimRequest(dim, dim_type)

        # read dimensions and allocate fresh dimensions as necessary
        for name, dim_request in name_to_dim.items():
            name_to_dim[name] = _DIM_STACK.request(name, dim_request)[1]

        msg["stop"] = True  # only need to run this once per to_data call

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_funsor(msg):

        raw_value, output = msg["args"]
        dim_to_name = msg["kwargs"].setdefault("dim_to_name", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        event_dim = len(output.shape)
        batch_dim = len(raw_value.shape) - event_dim

        # interpret all names/dims as requests since we only run this function once
        for dim in range(-batch_dim, 0):
            if raw_value.shape[dim - event_dim] == 1:
                continue
            name = dim_to_name.get(dim, None)
            dim_to_name[dim] = name if isinstance(name, NameRequest) else NameRequest(name, dim_type)

        for dim, name_request in dim_to_name.items():
            dim_to_name[dim] = _DIM_STACK.request(name_request, dim)[0]

        msg["stop"] = True  # only need to run this once per to_funsor call


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
        self._iter_parents = ()
        super().__init__()

    def generator(self, iterable):
        self._iterable = iterable
        return self

    def _get_iter_parents(self, frame):
        iter_parents = [frame]
        frontier = (frame,)
        while frontier:
            frontier = sum([p.iter_parents for p in frontier], ())
            iter_parents += frontier
        return tuple(iter_parents)

    def __iter__(self):
        assert self._iterable is not None
        self._iter_parents = self._get_iter_parents(_DIM_STACK.current_frame)
        with ExitStack() as stack:
            for value in self._iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        if self.keep and self._saved_frames:
            saved_frame = self._saved_frames.pop()
            name_to_dim, dim_to_name = saved_frame.name_to_dim, saved_frame.dim_to_name
        else:
            name_to_dim, dim_to_name = OrderedDict(), OrderedDict()

        frame = StackFrame(
            name_to_dim=name_to_dim, dim_to_name=dim_to_name,
            parents=tuple(reversed(_DIM_STACK._stack[len(_DIM_STACK._stack) - self.history:])),
            iter_parents=tuple(self._iter_parents),
            keep=self.keep
        )

        _DIM_STACK.push(frame)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self.keep:
            # don't keep around references to other frames
            old_frame = _DIM_STACK.pop()
            saved_frame = StackFrame(
                name_to_dim=old_frame.name_to_dim, dim_to_name=old_frame.dim_to_name,
                parents=(), iter_parents=(), keep=self.keep
            )
            self._saved_frames.append(saved_frame)
        else:
            _DIM_STACK.pop()
        return super().__exit__(*args, **kwargs)


class GlobalNamedMessenger(NamedMessenger):

    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        self._saved_globals = ()
        self._extant_globals = ()
        super().__init__()

    def __enter__(self):
        if self._ref_count == 0:
            if self.first_available_dim is not None:
                self._prev_first_dim = _DIM_STACK.set_first_available_dim(self.first_available_dim)
            self._extant_globals = frozenset(_DIM_STACK.global_frame.name_to_dim)
            for name, dim in self._saved_globals:
                _DIM_STACK.global_frame.write(name, dim)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1:
            if self.first_available_dim is not None:
                _DIM_STACK.set_first_available_dim(self._prev_first_dim)
            for name in frozenset(_DIM_STACK.global_frame.name_to_dim) - self._extant_globals:
                dim = _DIM_STACK.global_frame.name_to_dim[name]
                self._saved_globals += (_DIM_STACK.global_frame.free(name, dim),)
        return super().__exit__(*args, **kwargs)
