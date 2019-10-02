import importlib
from contextlib import contextmanager


_BACKEND = 'pyro'


class GenericModule(object):
    """
    Wrapper for a module that can be dynamically routed to a custom backend.
    """
    backend_name = None
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
    global _BACKEND
    old_backend_name = _BACKEND
    if aliases:
        assert len(aliases) == 1
        assert not new_backends
        new_backend_name = aliases[0]
        new_backends = _ALIASES[aliases[0]]
    else:
        new_backend_name = None

    old_backends = {}
    for name, new_backend in new_backends.items():
        old_backends[name] = GenericModule.current_backend[name]
        GenericModule.current_backend[name] = new_backend
    try:
        _BACKEND = new_backend_name
        yield
    finally:
        _BACKEND = old_backend_name
        for name, old_backend in old_backends.items():
            GenericModule.current_backend[name] = old_backend


_ALIASES = {
    'pyro': {
        'distributions': 'pyro.distributions',
        'handlers': 'pyro.poutine',
        'infer': 'pyro.infer',
        'ops': 'torch',
        'optim': 'pyro.optim',
        'pyro': 'pyro',
    },
    'minipyro': {
        'distributions': 'pyro.distributions',
        'handlers': 'pyro.poutine',
        'infer': 'pyro.contrib.minipyro',
        'ops': 'torch',
        'optim': 'pyro.contrib.minipyro',
        'pyro': 'pyro.contrib.minipyro',
    },
    'funsor': {
        'distributions': 'funsor.distributions',
        'handlers': 'pyro.poutine',
        'infer': 'funsor.minipyro',
        'ops': 'funsor.ops',
        'optim': 'funsor.minipyro',
        'pyro': 'funsor.minipyro',
    },
    'numpy': {
        'distributions': 'numpyro.compat.distributions',
        'handlers': 'numpyro.compat.handlers',
        'infer': 'numpyro.compat.infer',
        'ops': 'numpyro.compat.ops',
        'optim': 'numpyro.compat.optim',
        'pyro': 'numpyro.compat.pyro',
    },
}

# These modules can be overridden.
pyro = GenericModule('pyro', 'pyro')
distributions = GenericModule('distributions', 'pyro.distributions')
handlers = GenericModule('handlers', 'pyro.poutine')
infer = GenericModule('infer', 'pyro.infer')
optim = GenericModule('optim', 'pyro.optim')
ops = GenericModule('ops', 'torch')
