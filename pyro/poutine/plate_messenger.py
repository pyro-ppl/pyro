from __future__ import absolute_import, division, print_function

from .indep_messenger import IndepMessenger
from .runtime import _DIM_ALLOCATOR


class PlateMessenger(IndepMessenger):
    """
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
        super(PlateMessenger, self).__init__(name, size, dim)
        self.sites = sites
        self._allocated = False

    def __enter__(self):
        # XXX just defined this to return so it passes tests
        super(PlateMessenger, self).__enter__()
        return [i for i in range(self.size)] if self.size is not None else self

    def __exit__(self, *args):
        if self._allocated:
            _DIM_ALLOCATOR.free(self.name, self.dim)
            self._allocated = False
        return super(PlateMessenger, self).__exit__(*args)

    def _process_message(self, msg):
        if self.sites is None or msg["name"] in self.sites:
            if not self._allocated:
                self.dim = _DIM_ALLOCATOR.allocate(self.name, self.dim)
                self._allocated = True
            super(PlateMessenger, self)._process_message(msg)
        elif self.sites is not None and msg["name"] not in self.sites:
            if self._allocated:
                _DIM_ALLOCATOR.free(self.name, self.dim)
                self._allocated = False
