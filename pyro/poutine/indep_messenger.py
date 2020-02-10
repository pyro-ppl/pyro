# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numbers
from collections import namedtuple

import torch

from pyro.util import ignore_jit_warnings
from .messenger import Messenger
from .runtime import _DIM_ALLOCATOR


class CondIndepStackFrame(namedtuple("CondIndepStackFrame", ["name", "dim", "size", "counter"])):
    @property
    def vectorized(self):
        return self.dim is not None

    def _key(self):
        with ignore_jit_warnings(["Converting a tensor to a Python number"]):
            size = self.size.item() if isinstance(self.size, torch.Tensor) else self.size
            return self.name, self.dim, size, self.counter

    def __eq__(self, other):
        return type(self) == type(other) and self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        return self.name


class IndepMessenger(Messenger):
    """
    This messenger keeps track of stack of independence information declared by
    nested ``plate`` contexts. This information is stored in a
    ``cond_indep_stack`` at each sample/observe site for consumption by
    ``TraceMessenger``.

    Example::

        x_axis = IndepMessenger('outer', 320, dim=-1)
        y_axis = IndepMessenger('inner', 200, dim=-2)
        with x_axis:
            x_noise = sample("x_noise", dist.Normal(loc, scale).expand_by([320]))
        with y_axis:
            y_noise = sample("y_noise", dist.Normal(loc, scale).expand_by([200, 1]))
        with x_axis, y_axis:
            xy_noise = sample("xy_noise", dist.Normal(loc, scale).expand_by([200, 320]))

    """
    def __init__(self, name=None, size=None, dim=None, device=None):
        if not torch._C._get_tracing_state() and size == 0:
            raise ZeroDivisionError("size cannot be zero")

        super().__init__()
        self._vectorized = None
        if dim is not None:
            self._vectorized = True

        self._indices = None
        self.name = name
        self.dim = dim
        self.size = size
        self.device = device
        self.counter = 0

    def next_context(self):
        """
        Increments the counter.
        """
        self.counter += 1

    def __enter__(self):
        if self._vectorized is not False:
            self._vectorized = True

        if self._vectorized is True:
            self.dim = _DIM_ALLOCATOR.allocate(self.name, self.dim)

        return super().__enter__()

    def __exit__(self, *args):
        if self._vectorized is True:
            _DIM_ALLOCATOR.free(self.name, self.dim)
        return super().__exit__(*args)

    def __iter__(self):
        if self._vectorized is True or self.dim is not None:
            raise ValueError(
                "cannot use plate {} as both vectorized and non-vectorized"
                "independence context".format(self.name))

        self._vectorized = False
        self.dim = None
        with ignore_jit_warnings([("Iterating over a tensor", RuntimeWarning)]):
            for i in self.indices:
                self.next_context()
                with self:
                    yield i if isinstance(i, numbers.Number) else i.item()

    def _reset(self):
        if self._vectorized:
            _DIM_ALLOCATOR.free(self.name, self.dim)
        self._vectorized = None
        self.counter = 0

    @property
    def indices(self):
        if self._indices is None:
            self._indices = torch.arange(self.size, dtype=torch.long).to(self.device)
        return self._indices

    def _process_message(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, self.counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
