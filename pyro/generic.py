from __future__ import absolute_import, division, print_function

import importlib

from contextlib2 import contextmanager


class GenericModule(object):
    """
    Wrapper for a module that can be dynamically routed to a custom backend.
    """
    current_backend = {}
    _modules = {}

    def __init__(self, name, default_backend):
        assert isinstance(name, str)
        assert isinstance(default_backend, str)
        self._name = name
        GenericModule.current_backend[name] = default_backend

    def __getattribute__(self, name):
        module_name = super(GenericModule, self).__getattribute__('_name')
        backend = GenericModule.current_backend[module_name]
        try:
            module = GenericModule._modules[backend]
        except KeyError:
            module = importlib.import_module(backend)
            GenericModule._modules[backend] = module
        return getattr(module, name)


@contextmanager
def pyro_backend(*aliases, **new_backends):
    """
    Context manager to set a custom backend for Pyro models.

    Backends can be specified either by name (for standard backends)
    or by providing a dict mapping module name to backend module name.
    Standard backends include: pyro, minipyro, funsor, and numpy.
    """
    if aliases:
        assert len(aliases) == 1
        assert not new_backends
        new_backends = _ALIASES[aliases[0]]

    old_backends = {}
    for name, new_backend in new_backends.items():
        old_backends[name] = GenericModule.current_backend[name]
        GenericModule.current_backend[name] = new_backend
    try:
        yield
    finally:
        for name, old_backend in old_backends.items():
            GenericModule.current_backend[name] = old_backend


_ALIASES = {
    'pyro': {
        'pyro': 'pyro',
        'distributions': 'pyro.distributions',
        'infer': 'pyro.infer',
        'optim': 'pyro.optim',
    },
    'minipyro': {
        'pyro': 'pyro.contrib.minipyro',
        'infer': 'pyro.contrib.minipyro',
        'optim': 'pyro.contrib.minipyro',
    },
    'funsor': {
        'pyro': 'funsor.minipyro',
        'infer': 'funsor.minipyro',
        'optim': 'funsor.minipyro',
        'distributions': 'funsor.distributions',
    },
    'numpy': {
        'pyro': 'numpyro.compat.pyro',
        'distributions': 'numpyro.compat.distributions',
        'infer': 'numpyro.compat.infer',
        'optim': 'numpyro.compat.optim',
    },
}

# These modules can be overridden.
pyro = GenericModule('pyro', 'pyro')
distributions = GenericModule('distributions', 'pyro.distributions')
infer = GenericModule('infer', 'pyro.infer')
optim = GenericModule('optim', 'pyro.optim')
