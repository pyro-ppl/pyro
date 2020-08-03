# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter, OrderedDict, namedtuple
from enum import Enum


class StackFrame(object):
    """
    Consistent bidirectional mapping between integer positional dimensions and names.
    Can be queried like a dictionary (value = frame[key], frame[key] = value).
    """
    def __init__(self, name_to_dim, dim_to_name, history, keep):
        self.name_to_dim = name_to_dim
        self.dim_to_name = dim_to_name
        self.history = history
        self.keep = keep
        self.parent = None
        self.iter_parent = None

    def __setitem__(self, key, value):
        assert isinstance(key, (int, str)) and isinstance(value, (int, str)) and type(key) != type(value)
        name, dim = (value, key) if isinstance(key, int) else (key, value)
        self.name_to_dim[name], self.dim_to_name[dim] = dim, name

    def __getitem__(self, key):
        assert isinstance(key, (int, str))
        return self.dim_to_name[key] if isinstance(key, int) else self.name_to_dim[key]

    def __contains__(self, key):
        assert isinstance(key, (int, str))
        return key in (self.dim_to_name if isinstance(key, int) else self.name_to_dim)

    def pop(self, key):
        k2v, v2k = (self.dim_to_name, self.name_to_dim) if isinstance(key, int) else \
            (self.name_to_dim, self.dim_to_name)
        value = k2v.pop(key)
        key = v2k.pop(value)
        return value

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


DimRequest = namedtuple('DimRequest', ['value', 'dim_type'])
DimRequest.__new__.__defaults__ = (None, DimType.LOCAL)


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

    @property
    def current_write_env(self):
        return [self.current_frame] + (list(self.current_frame.parents) if self.current_frame.keep else [])

    @property
    def current_env(self):
        """
        Collect all frames necessary to compute the full name <--> dim mapping
        and interpret Funsor inputs or batch shapes at any point in a computation.
        """
        return [self.global_frame] + [self.current_frame] + \
            list(self.current_frame.parents) + list(self.current_frame.iter_parents)

    def _genvalue(self, key, value_request):
        """
        Given proposed values for a fresh (name, dim) pair, computes a new, possibly
        identical (name, dim) pair consistent with the current name <--> dim mapping.
        This function is pure and does not update the name <--> dim mapping itself.

        The implementation here is only one of several possibilities, and was chosen
        to match the behavior of Pyro's old enumeration machinery as closely as possible.
        """
        if isinstance(key, int):
            if value_request.value is None:
                fresh_value = "_pyro_dim_{}".format(-key)
            else:
                fresh_value = value_request.value
            return key, fresh_value

        elif isinstance(key, str):
            name, dim, dim_type = key, value_request.value, value_request.dim_type
            if dim is None:
                fresh_dim = self._first_available_dim if dim_type != DimType.VISIBLE else -1
                fresh_dim = -1 if fresh_dim is None else fresh_dim
            else:
                fresh_dim = value_request.value

            while any(fresh_dim in p for p in self.current_env):
                fresh_dim -= 1

            if fresh_dim < self.MAX_DIM or \
                    (dim_type == DimType.VISIBLE and fresh_dim <= self._first_available_dim):
                raise ValueError("Ran out of free dims during allocation for {}".format(name))

            return key, fresh_dim
        raise ValueError("{} and {} not a valid name-dim pair".format(key, value_request))

    def allocate(self, key_to_value_request):

        # step 1: fill in non-fresh
        for key, value_request in key_to_value_request.items():
            value = value_request.value
            for frame in self.current_env:
                found = (value is None and key in frame) or (value is not None and value in frame)
                if not found:
                    continue
                elif value is None and key in frame:
                    key_to_value_request[key] = frame[key]
                    break
                elif value is not None and frame[value] == key:
                    key_to_value_request[key] = value
                    break
                elif value is not None and frame[value] != key:
                    key_to_value_request[key] = value
                    break
                else:
                    raise ValueError("should not be here!")

        # step 2: split into fresh (key_to_value_request) and nonfresh (key_to_value)
        key_to_value = OrderedDict(
            (key, key_to_value_request.pop(key)) for key, value_request in tuple(key_to_value_request.items())
            if not isinstance(value_request, DimRequest)
        )

        # step 3: check for conflicts in non-fresh
        if max(Counter(key_to_value.values()).values(), default=0) > 1:
            raise ValueError("{} is not a valid shape request".format(key_to_value))

        # step 4: if no conflicts in non-fresh, allocate fresh values for all fresh
        for key, value_request in key_to_value_request.items():
            key, fresh_value = self._genvalue(key, value_request)
            value, fresh_key = self._genvalue(fresh_value, DimRequest(key, value_request.dim_type))
            for frame in [self.global_frame] if value_request.dim_type != DimType.LOCAL else self.current_write_env:
                frame[fresh_key] = fresh_value
            key_to_value[key] = fresh_value

        assert not any(isinstance(value, DimRequest) for value in key_to_value.values())
        return key_to_value

    def names_from_batch_shape(self, batch_shape, dim_type=DimType.LOCAL):
        return self.allocate_dim_to_name(OrderedDict(
            (dim, DimRequest(None, dim_type))
            for dim in range(-len(batch_shape), 0) if batch_shape[dim] > 1
        ))


_DIM_STACK = DimStack()  # only one global instance
