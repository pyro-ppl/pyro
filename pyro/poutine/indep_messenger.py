from __future__ import absolute_import, division, print_function

from collections import namedtuple

import torch

from .messenger import Messenger


class CondIndepStackFrame(namedtuple("CondIndepStackFrame", ["name", "dim", "size", "counter"])):
    @property
    def vectorized(self):
        return self.dim is not None

    def _key(self):
        size = self.size.item() if isinstance(self.size, torch.Tensor) else self.size
        return self.name, self.dim, size, self.counter

    def __eq__(self, other):
        return type(self) == type(other) and self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._key())


class IndepMessenger(Messenger):
    """
    This messenger keeps track of stack of independence information declared by
    nested ``irange`` and ``iarange`` contexts. This information is stored in
    a ``cond_indep_stack`` at each sample/observe site for consumption by
    ``TraceMessenger``.
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
