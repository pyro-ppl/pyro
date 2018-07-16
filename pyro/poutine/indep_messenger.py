from __future__ import absolute_import, division, print_function

from collections import namedtuple

from .messenger import Messenger
from .runtime import _DIM_ALLOCATOR


class CondIndepStackFrame(namedtuple("CondIndepStackFrame", ["name", "dim", "size", "counter"])):
    @property
    def vectorized(self):
        return self.dim is not None


class IndepMessenger(Messenger):
    """
    This messenger keeps track of stack of independence information declared by
    nested ``irange`` and ``iarange`` contexts. This information is stored in
    a ``cond_indep_stack`` at each sample/observe site for consumption by
    ``TraceMessenger``.

    Allows specifying iarange contexts outside of a model

    Example::

        @plate(name="outer", sites=["x_noise", "xy_noise"], size=320, dim=-1)
        @plate(name="inner", sites=["y_noise", "xy_noise"], size=200, dim=-2)
        def model():
            x_noise = sample("x_noise", dist.Normal(0., 1.).expand_by([320]))
            y_noise = sample("y_noise", dist.Normal(0., 1.).expand_by([200, 1]))
            xy_noise = sample("xy_noise", dist.Normal(0., 1.).expand_by([200, 320]))

    Example::

        x_axis = plate('outer', 320, dim=-1)
        y_axis = plate('inner', 200, dim=-2)
        with x_axis:
            x_noise = sample("x_noise", dist.Normal(loc, scale).expand_by([320]))
        with y_axis:
            y_noise = sample("y_noise", dist.Normal(loc, scale).expand_by([200, 1]))
        with x_axis, y_axis:
            xy_noise = sample("xy_noise", dist.Normal(loc, scale).expand_by([200, 320]))

    """
    def __init__(self, name=None, size=None, dim=None, sites=None):
        super(IndepMessenger, self).__init__()
        self.sites = sites
        self._installed = False
        self._vectorized = True
        if size == 0:
            # XXX hack to pass poutine tests
            raise ZeroDivisionError("size cannot be zero")
        self.name = name
        self.dim = dim
        self.size = size
        self.counter = 0

    def next_context(self):
        """
        Increments the counter.
        """
        self.counter += 1

    def __exit__(self, *args):
        if self._installed:
            if self._vectorized:
                _DIM_ALLOCATOR.free(self.name, self.dim)
            self._installed = False
        # self.counter = 0
        return super(IndepMessenger, self).__exit__(*args)

    def _reset(self):
        if self._installed:
            if self._vectorized:
                _DIM_ALLOCATOR.free(self.name, self.dim)
        self._installed = False
        self._vectorized = True
        self.counter = 0

    def _process_message(self, msg):
        if self.sites is None or msg["name"] in self.sites:
            if not self._installed:
                if self._vectorized:
                    self.dim = _DIM_ALLOCATOR.allocate(self.name, self.dim)
                self._installed = True
            frame = CondIndepStackFrame(self.name, self.dim, self.size, self.counter)
            msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        elif self.sites is not None and msg["name"] not in self.sites:
            if self._installed:
                if self._vectorized:
                    _DIM_ALLOCATOR.free(self.name, self.dim)
                self._installed = False
