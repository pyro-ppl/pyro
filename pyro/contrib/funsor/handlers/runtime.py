# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
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

    def write_frames(self, dim_type):
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

    def request(self, name, dim):
        """
        Given proposed, possibly empty values of a (name, dim) pair, this function
        attempts to fill in the values according to the current name <--> dim mapping
        and updates the global DimStack's state to reflect the result.
        """
        assert isinstance(name, NameRequest) ^ isinstance(dim, DimRequest)
        if isinstance(dim, DimRequest):
            dim, dim_type = dim.dim, dim.dim_type
        elif isinstance(name, NameRequest):
            name, dim_type = name.name, name.dim_type

        # read dimension
        for frame in self.current_env:
            name, dim, found = frame.read(name, dim)
            if found:
                break

        # generate fresh name or dimension
        if not found:
            name, dim = self._gendim(NameRequest(name, dim_type), DimRequest(dim, dim_type))
            # store the fresh dimension
            for frame in self.write_frames(dim_type):
                frame.write(name, dim)

        return name, dim

    def names_from_batch_shape(self, batch_shape, dim_type=DimType.LOCAL):
        dim_to_name = OrderedDict()
        for dim in range(-len(batch_shape), 0):
            if batch_shape[dim] == 1:
                continue
            dim_to_name[dim] = self.request(NameRequest(None, dim_type), dim)[0]
        return dim_to_name


_DIM_STACK = DimStack()  # only one global instance
