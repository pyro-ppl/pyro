# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from contextlib import ExitStack

from pyro.poutine.reentrant_messenger import ReentrantMessenger

from pyro.contrib.funsor.handlers.runtime import _DIM_STACK, DimRequest, DimType, NameRequest, StackFrame


class DimStackCleanupMessenger(ReentrantMessenger):

    def __init__(self):
        self._saved_dims = ()
        return super().__init__()

    def __enter__(self):
        if self._ref_count == 0 and _DIM_STACK.outermost is None:
            _DIM_STACK.outermost = self
            for name, dim in self._saved_dims:
                _DIM_STACK.global_frame.write(name, dim)
            self._saved_dims = ()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1 and _DIM_STACK.outermost is self:
            _DIM_STACK.outermost = None
            _DIM_STACK.set_first_available_dim(-1)
            for name, dim in reversed(tuple(_DIM_STACK.global_frame.name_to_dim.items())):
                self._saved_dims += (_DIM_STACK.global_frame.free(name, dim),)
        return super().__exit__(*args, **kwargs)


class NamedMessenger(DimStackCleanupMessenger):

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_data(msg):

        funsor_value, = msg["args"]
        name_to_dim = msg["kwargs"].setdefault("name_to_dim", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        batch_names = tuple(funsor_value.inputs.keys())

        # interpret all names/dims as requests since we only run this function once
        for name in batch_names:
            dim = name_to_dim.get(name, None)
            name_to_dim[name] = _DIM_STACK.request(
                name, dim if isinstance(dim, DimRequest) else DimRequest(dim, dim_type))[1]

        msg["stop"] = True  # only need to run this once per to_data call

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_funsor(msg):

        if len(msg["args"]) == 2:
            raw_value, output = msg["args"]
        else:
            raw_value = msg["args"][0]
            output = msg["kwargs"].setdefault("output", None)
        dim_to_name = msg["kwargs"].setdefault("dim_to_name", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        event_dim = len(output.shape) if output else 0
        try:
            batch_shape = raw_value.batch_shape  # TODO make make this more robust
        except AttributeError:
            full_shape = getattr(raw_value, "shape", ())
            batch_shape = full_shape[:len(full_shape) - event_dim]

        # interpret all names/dims as requests since we only run this function once
        for dim in range(-len(batch_shape), 0):
            if batch_shape[dim] == 1:
                continue
            name = dim_to_name.get(dim, None)
            dim_to_name[dim] = _DIM_STACK.request(
                name if isinstance(name, NameRequest) else NameRequest(name, dim_type), dim)[0]

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

    def __call__(self, fn):
        if fn is not None and not callable(fn):
            self._iterable = fn
            return self
        return super().__call__(fn)

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

    def __init__(self):
        self._saved_globals = ()
        super().__init__()

    def __enter__(self):
        if self._ref_count == 0:
            for name, dim in self._saved_globals:
                _DIM_STACK.global_frame.write(name, dim)
            self._saved_globals = ()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1:
            for name, dim in self._saved_globals:
                _DIM_STACK.global_frame.free(name, dim)
        return super().__exit__(*args, **kwargs)

    def _pyro_post_to_funsor(self, msg):
        if msg["kwargs"]["dim_type"] in (DimType.GLOBAL, DimType.VISIBLE):
            for name in msg["value"].inputs:
                self._saved_globals += ((name, _DIM_STACK.global_frame.name_to_dim[name]),)

    def _pyro_post_to_data(self, msg):
        if msg["kwargs"]["dim_type"] in (DimType.GLOBAL, DimType.VISIBLE):
            for name in msg["args"][0].inputs:
                self._saved_globals += ((name, _DIM_STACK.global_frame.name_to_dim[name]),)


class BaseEnumMessenger(NamedMessenger):
    """
    handles first_available_dim management, enum effects should inherit from this
    """
    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        super().__init__()

    def __enter__(self):
        if self._ref_count == 0 and self.first_available_dim is not None:
            self._prev_first_dim = _DIM_STACK.set_first_available_dim(self.first_available_dim)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1 and self.first_available_dim is not None:
            _DIM_STACK.set_first_available_dim(self._prev_first_dim)
        return super().__exit__(*args, **kwargs)
