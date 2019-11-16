import functools

import torch
from pyro.nn.module import PyroModule

import pyro.distributions.torch
import pyro.distributions.torch_distribution


def dist_as_module(Dist):
    """
    Dynamically create a subclass of ``(Dist, PyroModule)``.

    This allows trainable parameters by creating a proxy ``Dist`` object with
    fresh parameters on call of each method.
    """
    assert isinstance(Dist, type)
    assert issubclass(Dist, torch.distributions.Distribution)
    if issubclass(Dist, PyroModule):
        return Dist
    if Dist in dist_as_module._cache:
        return dist_as_module._cache[Dist]

    class result(Dist, PyroModule):
        # TODO convert *args to **kwargs as in funsor
        def __init__(self, **kwargs):
            self._pyro_init_keys = frozenset(kwargs)
            self._pyro_active = 0
            self._pyro_proxy = None

            # Intentionally skip Dist.__init__().
            proxy = Dist(**kwargs)
            torch.distributions.Distribution.__init__(
                self,
                proxy.batch_shape,
                proxy.event_shape,
                proxy._validate_args,
            )

            # Store kwargs in self as PyroModule.
            for key, value in kwargs.items():
                if (isinstance(value, torch.Tensor) and
                        not isinstance(value, torch.nn.Parameter)):
                    self.register_buffer(key, value)
                else:
                    setattr(self, key, value)

        def _pyro_enter(self):
            self._pyro_active += 1
            if self._pyro_active == 1:
                self._pyro_proxy = Dist(**{key: getattr(self, key)
                                           for key in self._pyro_init_keys})
            return self._pyro_proxy

        def _pyro_exit(self):
            self._pyro_active -= 1
            if self._pyro_active == 0:
                self._pyro_proxy = None

        def to_event(self, reinterpreted_batch_ndims=None):
            if reinterpreted_batch_ndims is None:
                reinterpreted_batch_ndims = len(self.batch_shape)
            Independent = dist_as_module(pyro.distributions.torch.Independent)
            return Independent(self, reinterpreted_batch_ndims)

        def mask(self, mask):
            MaskedDistribution = dist_as_module(
                pyro.distributions.torch_distribution.MaskedDistribution)
            return MaskedDistribution(self, mask)

        # Unpickling helper to load an object of type dist_as_module(Dist).
        def __reduce__(self):
            state = getattr(self, '__getstate__', self.__dict__.copy)()
            return _New, (Dist,), state

    for name in dir(Dist):
        if not name.startswith('_') and name not in ('shape', 'to_event', 'mask'):
            if callable(getattr(Dist, name)):
                setattr(result, name, _proxied(getattr(Dist, name)))

    result.__name__ = Dist.__name__ + "PyroModule"
    dist_as_module._cache[Dist] = result
    return result


dist_as_module._cache = {}


def _proxied(method):

    @functools.wraps(method)
    def proxied_method(self, *args, **kwargs):
        try:
            proxy = self._pyro_enter()
            return method(proxy, *args, **kwargs)
        finally:
            self._pyro_exit()

    return proxied_method


# Unpickling helper to create an empty object of type dist_as_module[Dist].
class _New:
    def __init__(self, Dist):
        self.__class__ = dist_as_module[Dist]
