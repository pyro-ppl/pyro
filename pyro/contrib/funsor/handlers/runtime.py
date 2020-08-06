# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter, OrderedDict, namedtuple
from enum import Enum


class StackFrame:
    """
    Consistent bidirectional mapping between integer positional dimensions and names.
    Can be queried like a dictionary (value = frame[key], frame[key] = value).
    """
    def __init__(self, name_to_dim, dim_to_name, history=1, keep=False):
        assert isinstance(name_to_dim, OrderedDict) and \
            all(isinstance(name, str) and isinstance(dim, int) for name, dim in name_to_dim.items())
        assert isinstance(dim_to_name, OrderedDict) and \
            all(isinstance(name, str) and isinstance(dim, int) for dim, name in dim_to_name.items())
        self.name_to_dim = name_to_dim
        self.dim_to_name = dim_to_name
        self.history = history
        self.keep = keep

    def __setitem__(self, key, value):
        assert isinstance(key, (int, str)) and isinstance(value, (int, str)) and type(key) != type(value)
        name, dim = (value, key) if isinstance(key, int) else (key, value)
        self.name_to_dim[name], self.dim_to_name[dim] = dim, name

    def __getitem__(self, key):
        assert isinstance(key, (int, str))
        return self.dim_to_name[key] if isinstance(key, int) else self.name_to_dim[key]

    def __delitem__(self, key):
        assert isinstance(key, (int, str))
        k2v, v2k = (self.dim_to_name, self.name_to_dim) if isinstance(key, int) else \
            (self.name_to_dim, self.dim_to_name)
        del v2k[k2v[key]]
        del k2v[key]

    def __contains__(self, key):
        assert isinstance(key, (int, str))
        return key in (self.dim_to_name if isinstance(key, int) else self.name_to_dim)


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
        global_frame = StackFrame(
            name_to_dim=OrderedDict(), dim_to_name=OrderedDict(),
            history=0, keep=False,
        )
        self._local_stack = [global_frame]
        self._iter_stack = [global_frame]
        self._global_stack = [global_frame]
        self._first_available_dim = self.DEFAULT_FIRST_DIM
        self.outermost = None

    MAX_DIM = -25
    DEFAULT_FIRST_DIM = -5

    def set_first_available_dim(self, dim):
        assert dim is None or (self.MAX_DIM < dim < 0)
        old_dim, self._first_available_dim = self._first_available_dim, dim
        return old_dim

    def push_global(self, frame):
        self._global_stack.append(frame)

    def pop_global(self):
        assert self._global_stack, "cannot pop the global frame"
        return self._global_stack.pop()

    def push_iter(self, frame):
        self._iter_stack.append(frame)

    def pop_iter(self):
        assert self._iter_stack, "cannot pop the global frame"
        return self._iter_stack.pop()

    def push_local(self, frame):
        self._local_stack.append(frame)

    def pop_local(self):
        assert self._local_stack, "cannot pop the global frame"
        return self._local_stack.pop()

    @property
    def global_frame(self):
        return self._global_stack[-1]

    @property
    def local_frame(self):
        return self._local_stack[-1]

    @property
    def current_write_env(self):
        return self._local_stack[-1:] if not self.local_frame.keep else \
            self._local_stack[-self.local_frame.history-1:]

    @property
    def current_read_env(self):
        """
        Collect all frames necessary to compute the full name <--> dim mapping
        and interpret Funsor inputs or batch shapes at any point in a computation.
        """
        return self._global_stack + self._local_stack[-self.local_frame.history-1:] + self._iter_stack

    def _genvalue(self, key, value_request):
        """
        Given proposed values for a fresh (name, dim) pair, computes a new, possibly
        identical (name, dim) pair consistent with the current name <--> dim mapping.
        This function is pure and does not update the name <--> dim mapping itself.

        The implementation here is only one of several possibilities, and was chosen
        to match the behavior of Pyro's old enumeration machinery as closely as possible.
        """
        if isinstance(key, int):
            dim, name = key, value_request.value
            fresh_name = "_pyro_dim_{}".format(-key) if name is None else name
            return dim, fresh_name

        elif isinstance(key, str):
            name, dim, dim_type = key, value_request.value, value_request.dim_type
            if dim_type == DimType.VISIBLE:
                fresh_dim = -1 if dim is None else dim
            else:
                fresh_dim = self._first_available_dim  # discard input...

            while any(fresh_dim in p for p in self.current_read_env):
                fresh_dim -= 1

            if fresh_dim < self.MAX_DIM or \
                    (dim_type == DimType.VISIBLE and fresh_dim <= self._first_available_dim):
                raise ValueError("Ran out of free dims during allocation for {}".format(name))

            return name, fresh_dim
        raise ValueError("{} and {} not a valid name-dim pair".format(key, value_request))

    def allocate(self, key_to_value_request):

        # step 1: split into fresh and non-fresh
        key_to_value = OrderedDict()
        for key, value_request in tuple(key_to_value_request.items()):
            value = value_request.value
            for frame in self.current_read_env:
                if value is None and key in frame:
                    key_to_value[key] = frame[key]
                    del key_to_value_request[key]
                    break
                elif value is not None and value in frame:
                    key_to_value[key] = value
                    del key_to_value_request[key]
                    break

        # step 2: check that the non-fresh input mapping from keys to values is 1-1
        if max(Counter(key_to_value.values()).values(), default=0) > 1:
            raise ValueError("{} is not a valid shape request".format(key_to_value))

        # step 3: allocate fresh values for all fresh
        for key, value_request in key_to_value_request.items():
            key, fresh_value = self._genvalue(key, value_request)
            # if this key is already active but inconsistent with the fresh value,
            # generate a fresh_key for future conversions via _genvalue in reverse
            if value_request.dim_type != DimType.VISIBLE or any(key in frame for frame in self.current_read_env):
                _, fresh_key = self._genvalue(fresh_value, DimRequest(key, value_request.dim_type))
            else:
                fresh_key = key
            for frame in ([self.global_frame] if value_request.dim_type != DimType.LOCAL else self.current_write_env):
                frame[fresh_key] = fresh_value
            # use the user-provided key rather than fresh_key for satisfying this request only
            key_to_value[key] = fresh_value

        assert not any(isinstance(value, DimRequest) for value in key_to_value.values())
        return key_to_value

    def names_from_batch_shape(self, batch_shape, dim_type=DimType.LOCAL):
        return self.allocate_dim_to_name(OrderedDict(
            (dim, DimRequest(None, dim_type))
            for dim in range(-len(batch_shape), 0) if batch_shape[dim] > 1
        ))


_DIM_STACK = DimStack()  # only one global instance
