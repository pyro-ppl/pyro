# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter, OrderedDict, namedtuple
from enum import Enum


class StackFrame(object):

    def __init__(self, name_to_dim, dim_to_name, history, keep):
        self.name_to_dim = name_to_dim
        self.dim_to_name = dim_to_name
        self.history = history
        self.keep = keep
        self.parent = None
        self.iter_parent = None

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

    @property
    def parents(self):
        parent = self
        for h in range(self.history):
            parent = parent.parent
            if parent is not None:
                yield parent
            else:
                break

    @property
    def iter_parents(self):
        iter_parent = self
        while iter_parent.iter_parent is not None:
            iter_parent = iter_parent.iter_parent
            yield iter_parent


class DimType(Enum):
    """Enumerates the possible types of dimensions to allocate"""
    LOCAL = 0
    GLOBAL = 1
    VISIBLE = 2


DimRequest = namedtuple('DimRequest', ['dim', 'dim_type'])
DimRequest.__new__.__defaults__ = (None, DimType.LOCAL)
NameRequest = namedtuple('NameRequest', ['name', 'dim_type'])
NameRequest.__new__.__defaults__ = (None, DimType.LOCAL)


class DimStack:
    """
    Single piece of global state to keep track of the mapping between names and dimensions.

    Replaces the plate DimAllocator, the enum EnumAllocator, the stack in MarkovMessenger,
    _param_dims and _value_dims in EnumMessenger, and dim_to_symbol in msg['infer']
    """
    def __init__(self):
        self.global_frame = StackFrame(
            name_to_dim=OrderedDict(), dim_to_name=OrderedDict(),
            history=0, keep=False,
        )
        self.current_frame = self.global_frame
        self.iter_frame = None
        self._first_available_dim = self.DEFAULT_FIRST_DIM
        self.outermost = None

    MAX_DIM = -25
    DEFAULT_FIRST_DIM = -5

    def set_first_available_dim(self, dim):
        assert dim is None or (self.MAX_DIM < dim < 0)
        old_dim, self._first_available_dim = self._first_available_dim, dim
        return old_dim

    def push(self, frame):
        frame.parent, frame.iter_parent = self.current_frame, self.iter_frame
        self.current_frame = frame

    def pop(self):
        assert self.current_frame.parent is not None, "cannot pop the global frame"
        popped_frame = self.current_frame
        self.current_frame = popped_frame.parent
        # don't keep around references to other frames
        popped_frame.parent, popped_frame.iter_parent = None, None
        return popped_frame

    def get_current_write_frames(self, dim_type):
        if dim_type == DimType.LOCAL:
            return [self.current_frame] + \
                (list(self.current_frame.parents) if self.current_frame.keep else [])
        else:
            return [self.global_frame]

    @property
    def current_env(self):
        """
        Collect all frames necessary to compute the full name <--> dim mapping
        and interpret Funsor inputs or batch shapes at any point in a computation.
        """
        return [self.global_frame] + [self.current_frame] + \
            list(self.current_frame.parents) + list(self.current_frame.iter_parents)

    def _gendim(self, name_request, dim_request):
        """
        Given proposed values for a fresh (name, dim) pair, computes a new, possibly
        identical (name, dim) pair consistent with the current name <--> dim mapping.
        This function is pure and does not update the name <--> dim mapping itself.

        The implementation here is only one of several possibilities, and was chosen
        to match the behavior of Pyro's old enumeration machinery as closely as possible.
        """
        assert isinstance(name_request, NameRequest) and isinstance(dim_request, DimRequest)
        dim_type = dim_request.dim_type

        if name_request.name is None:
            fresh_name = "_pyro_dim_{}".format(-dim_request.dim)
        else:
            fresh_name = name_request.name

        conflict_frames = self.current_env
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
            raise ValueError("Ran out of free dims during allocation for {}".format(fresh_name))

        return fresh_name, fresh_dim

    def allocate_dim_to_name(self, dim_to_name_request):

        # step 1: fill in non-fresh
        for dim, name_request in dim_to_name_request.items():
            name = name_request.name
            for frame in self.current_env:
                found = (name is None and dim in frame.dim_to_name) or \
                    (name is not None and name in frame.name_to_dim)
                if not found:
                    continue
                elif name is None and dim in frame.dim_to_name:
                    dim_to_name_request[dim] = frame.dim_to_name[dim]
                    break
                elif name is not None and frame.name_to_dim[name] == dim:
                    dim_to_name_request[dim] = name
                    break
                elif name is not None and frame.name_to_dim[name] != dim:
                    dim_to_name_request[dim] = name
                    break
                else:
                    raise ValueError("should not be here!")

        # step 2: split into fresh and non-fresh
        dim_to_fresh_name_request = OrderedDict(
            (dim, dim_to_name_request.pop(dim)) for dim, name_request in tuple(dim_to_name_request.items())
            if isinstance(name_request, NameRequest)  # and name_request.name is None
        )

        # step 3: check for conflicts in non-fresh
        if max(Counter(dim_to_name_request.values()).values(), default=0) > 1:
            raise ValueError("{} is not a valid dim_to_name".format(dim_to_name_request))

        # step 4: if no conflicts in non-fresh, allocate fresh dims for all fresh
        for dim, name_request in dim_to_fresh_name_request.items():
            name, fresh_dim = self._gendim(name_request, DimRequest(None, name_request.dim_type))
            for frame in self.get_current_write_frames(name_request.dim_type):
                frame.write(name, fresh_dim)
            dim_to_name_request[dim] = name

        assert all(isinstance(name, str) for name in dim_to_name_request.values())
        return dim_to_name_request

    def allocate_name_to_dim(self, name_to_dim_request):

        # step 1: fill in non-fresh
        for name, dim_request in name_to_dim_request.items():
            dim = dim_request.dim
            for frame in self.current_env:
                found = (dim is None and name in frame.name_to_dim) or \
                    (dim is not None and dim in frame.dim_to_name)
                if not found:
                    continue
                elif dim is None and name in frame.name_to_dim:
                    name_to_dim_request[name] = frame.name_to_dim[name]
                    break
                elif dim is not None and frame.dim_to_name[dim] == name:
                    name_to_dim_request[name] = dim
                    break
                elif dim is not None and frame.dim_to_name[dim] != name:
                    name_to_dim_request[name] = dim
                    break
                else:
                    raise ValueError("should not be here!")

        # step 2: split into fresh and non-fresh
        name_to_fresh_dim_request = OrderedDict(
            (name, name_to_dim_request.pop(name)) for name, dim_request in tuple(name_to_dim_request.items())
            if isinstance(dim_request, DimRequest)  # and dim_request.dim is None
        )

        # step 3: check for conflicts in non-fresh
        if max(Counter(name_to_dim_request.values()).values(), default=0) > 1:
            raise ValueError("{} is not a valid name_to_dim".format(name_to_dim_request))

        # step 4: if no conflicts in non-fresh, allocate fresh dims for all fresh
        for name, dim_request in name_to_fresh_dim_request.items():
            fresh_name, dim = self._gendim(NameRequest(name, dim_request.dim_type), dim_request)
            for frame in self.get_current_write_frames(dim_request.dim_type):
                frame.write(fresh_name, dim)
            name_to_dim_request[name] = dim

        assert all(isinstance(dim, int) for dim in name_to_dim_request.values())
        return name_to_dim_request

    def names_from_batch_shape(self, batch_shape, dim_type=DimType.LOCAL):
        return self.allocate_dim_to_name(OrderedDict(
            (dim, NameRequest(None, dim_type))
            for dim in range(-len(batch_shape), 0) if batch_shape[dim] > 1
        ))


_DIM_STACK = DimStack()  # only one global instance
