from __future__ import absolute_import, division, print_function

from collections import namedtuple

from .poutine import Messenger


class CondIndepStackFrame(namedtuple("CondIndepStackFrame", ["name", "dim", "size", "counter"])):
    @property
    def vectorized(self):
        return self.dim is not None


class _DimAllocator(object):
    """
    Dimension allocator for internal use by :class:`iarange`.
    There is a single global instance.

    Note that dimensions are indexed from the right, e.g. -1, -2.
    """
    def __init__(self):
        self._stack = []  # in reverse orientation of log_prob.shape

    def allocate(self, name, dim):
        """
        Allocate a dimension to an :class:`iarange` with given name.
        Dim should be either None for automatic allocation or a negative
        integer for manual allocation.
        """
        if name in self._stack:
            raise ValueError('duplicate iarange "{}"'.format(name))
        if dim is None:
            # Automatically allocate the rightmost dimension to the left of all existing dims.
            self._stack.append(name)
            dim = -len(self._stack)
        elif dim >= 0:
            raise ValueError('Expected dim < 0 to index from the right, actual {}'.format(dim))
        else:
            # Allocate the requested dimension.
            while dim < -len(self._stack):
                self._stack.append(None)
            if self._stack[-1 - dim] is not None:
                raise ValueError('\n'.join([
                    'at iaranges "{}" and "{}", collide at dim={}'.format(name, self._stack[-1 - dim], dim),
                    '\nTry moving the dim of one iarange to the left, e.g. dim={}'.format(dim - 1)]))
            self._stack[-1 - dim] = name
        return dim

    def free(self, name, dim):
        """
        Free a dimension.
        """
        assert self._stack[-1 - dim] == name
        self._stack[-1 - dim] = None
        while self._stack and self._stack[-1] is None:
            self._stack.pop()


_DIM_ALLOCATOR = _DimAllocator()


class IndepMessenger(Messenger):
    """
    This messenger keeps track of stack of independence information declared by
    nested ``irange`` and ``iarange`` contexts. This information is stored in
    a ``cond_indep_stack`` at each sample/observe site for consumption by
    ``TracePoutine``.
    """
    def __init__(self, name, size, dim=None):
        """
        Constructor: basically default, but store a counter to keep track of
        which ``irange`` branch we're in.
        """
        super(IndepMessenger, self).__init__()
        self.name = name
        self.dim = dim
        self.size = size
        self.counter = 0

    def next_context(self):
        """
        Increments the counter.
        """
        self.counter += 1

    def _process_message(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, self.counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        return None
