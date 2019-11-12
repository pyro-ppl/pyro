"""
Pyro includes a special class :class:`~pyro.nn.module.PyroModule` whose
attributes can be modified by Pyro effects. To create one of these attributes,
use either the :class:`PyroParam` struct or the :class:`PyroSample` struct::

    my_module = PyroModule()
    my_module.x = PyroParam(torch.tensor(1.), constraint=constraints.positive)
    my_module.y = PyroSample(dist.Normal(0, 1))

"""
from collections import OrderedDict, namedtuple

import torch
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist
from pyro.poutine.runtime import _PYRO_PARAM_STORE

PyroParam = namedtuple("PyroParam", ("init_value", "constraint", "event_dim"))
PyroParam.__new__.__defaults__ = (constraints.real, None)
PyroParam.__doc__ = """
Structure to declare a Pyro-managed learnable parameter of a :class:`PyroModule`.
"""

PyroSample = namedtuple("PyroSample", ("prior",))
PyroSample.__doc__ = """
Structure to declare a Pyro-managed random parameter of a :class:`PyroModule`.
"""


def _make_name(prefix, name):
    return "{}.{}".format(prefix, name) if prefix else name


def _force_pyro_param(name, unconstrained_value, constraint=constraints.real, event_dim=None):
    """
    Force updates the Pyro param store with an unconstrained value.

    This treats a PyroModule as source of truth, rather than the Pyro param store.
    """
    if _PYRO_PARAM_STORE._params.get(name) is not unconstrained_value:
        _PYRO_PARAM_STORE._params[name] = unconstrained_value
        _PYRO_PARAM_STORE._param_to_name[unconstrained_value] = name
        _PYRO_PARAM_STORE._constraints[name] = constraint
    return pyro.param(name, event_dim=event_dim)


class _Cache:
    """
    Sometimes-active cache for ``PyroModule.__call__()`` contexts.
    """
    def __init__(self):
        self.active = 0
        self.cache = {}

    def __enter__(self):
        self.active += 1

    def __exit__(self, type, value, traceback):
        self.active -= 1
        if not self.active:
            self.cache.clear()

    def get(self, name):
        if self.active:
            return self.cache.get(name)

    def set(self, name, value):
        if self.active:
            self.cache[name] = value


class PyroModule(torch.nn.Module):
    """
    Subclass of :class:`torch.nn.Module` that supports setting of
    :class:`PyroParam` and :class:`PyroSample` .

    To create a Pyro-managed parameter attribute, set that attribute using the
    :class:`PyroParam` helper. Reading that attribute will then trigger a
    :func:`pyro.param` statement. For example::

        # Create Pyro-managed parameter attributes.
        my_module = PyroModule()
        my_module.loc = PyroParam(torch.tensor(0.))
        my_module.scale = PyroParam(torch.tensor(1.),
                                    constraint=constraints.positive)
        # Read the attributes.
        loc = my_module.loc  # Triggers a pyro.param statement.
        scale = my_module.scale  # Triggers another pyro.param statement.

    Note that, unlike normal :class:`torch.nn.Module` s, :class:`PyroModule` s
    should not be registered with :func:`pyro.module` statements.
    :class:`PyroModule` s can contain normal :class:`torch.nn.Module` s, but not
    vice versa. Accessing a normal :class:`torch.nn.Module` attribute of
    a :class:`PyroModule` triggers a :func:`pyro.module` statement. Parameters
    in :class:`PyroModule` s are considered the source of truth, and override
    values in the Pyro param store.

    To create a Pyro-managed random attribute, set that attribute using the
    :class:`PyroSample` helper, specifying a prior distribution. Reading that
    attribute will then trigger a :func:`pyro.sample` statement. For example::

        # Create Pyro-managed random attributes.
        my_module.x = PyroSample(dist.Normal(0, 1))
        my_module.y = PyroSample(lambda self: dist.Normal(self.loc, self.scale))

        # Sample the attributes.
        x = my_module.x  # Triggers a pyro.sample statement.
        y = my_module.y  # Triggers one pyro.sample + two pyro.param statements.

    Note that param and sample attribute access is cached within each
    invocation of ``.__call__()``. Because sample statements can appear only
    once in a Pyro trace, you should ensure that traced access to sample
    attributes is wrapped in a single invocation of ``.__call__()``.

    To make an existing module probabilistic, you can create a subclass and
    overwrite some parameters with :class:`PyroSample` s::

        class RandomLinear(nn.Linear, PyroModule):
            def __init__(in_features, out_features):
                super().__init__(in_features, out_features)
                self.weight = PyroSample(
                    lambda self: dist.Normal(0, 1)
                                     .expand([self.out_features,
                                              self.in_features])
                                     .to_event(2))
    """
    def __init__(self):
        self._pyro_name = ""
        self._pyro_cache = _Cache()  # shared among sub-PyroModules
        self._pyro_params = OrderedDict()
        self._pyro_samples = OrderedDict()
        super().__init__()

    def _pyro_set_supermodule(self, name, cache):
        self._pyro_name = name
        self._pyro_cache = cache
        for key, value in self._modules.items():
            assert isinstance(value, PyroModule)
            value._pyro_set_supermodule(_make_name(name, key), cache)

    def __call__(self, *args, **kwargs):
        with self._pyro_cache:
            return super().__call__(*args, **kwargs)

    def _pyro_sample(self, name, fn):
        cache = self.__dict__['_pyro_cache']
        value = cache.get(name)
        if value is None:
            value = pyro.sample(name, fn)
            cache.set(name, value)
        return value

    def __getattr__(self, name):
        # PyroParams trigger pyro.param statements.
        if '_pyro_params' in self.__dict__:
            _pyro_params = self.__dict__['_pyro_params']
            if name in _pyro_params:
                constraint, event_dim = _pyro_params[name]
                unconstrained_value = getattr(self, name + "_unconstrained")
                return _force_pyro_param(_make_name(self._pyro_name, name),
                                         unconstrained_value, constraint, event_dim)

        # PyroSample trigger pyro.sample statements.
        if '_pyro_samples' in self.__dict__:
            _pyro_samples = self.__dict__['_pyro_samples']
            if name in _pyro_samples:
                prior = _pyro_samples[name]
                if not isinstance(prior, (dist.Distribution, torch.distributions.Distribution)):
                    prior = prior(self)
                return self._pyro_sample(_make_name(self._pyro_name, name), prior)

        result = super().__getattr__(name)

        # Regular nn.Parameters trigger pyro.param statements.
        if isinstance(result, torch.nn.Parameter) and not name.endswith("_unconstrained"):
            _force_pyro_param(_make_name(self._pyro_name, name), result)

        # Regular nn.Modules trigger pyro.module statements.
        if isinstance(result, torch.nn.Module) and not isinstance(result, PyroModule):
            pyro.module(_make_name(self._pyro_name, name), result)

        return result

    def __setattr__(self, name, value):
        if isinstance(value, PyroModule):
            # Create a new sub PyroModule, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            value._pyro_set_supermodule(_make_name(self._pyro_name, name), self._pyro_cache)
            super().__setattr__(name, value)
            return

        if isinstance(value, PyroParam):
            # Create a new PyroParam, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            fullname = _make_name(self._pyro_name, name)
            if fullname in _PYRO_PARAM_STORE:
                del _PYRO_PARAM_STORE[fullname]
            _pyro_params = self.__dict__['_pyro_params']
            constrained_value, constraint, event_dim = value
            _pyro_params[name] = constraint, event_dim
            pyro.param(fullname, constrained_value, constraint=constraint, event_dim=event_dim)
            unconstrained_param = _PYRO_PARAM_STORE._params[fullname]
            if not isinstance(unconstrained_param, torch.nn.Parameter):
                unconstrained_param = torch.nn.Parameter(unconstrained_param)
                _PYRO_PARAM_STORE._params[fullname] = unconstrained_param
                _PYRO_PARAM_STORE._param_to_name[unconstrained_param] = fullname
            super().__setattr__(name + "_unconstrained", unconstrained_param)
            return

        if isinstance(value, torch.nn.Parameter):
            # Create a new nn.Parameter, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            fullname = _make_name(self._pyro_name, name)
            if fullname in _PYRO_PARAM_STORE:
                del _PYRO_PARAM_STORE[fullname]
            pyro.param(fullname, value)
            super().__setattr__(name, value)
            return

        if isinstance(value, torch.Tensor):
            if '_pyro_params' in self.__dict__:
                _pyro_params = self.__dict__['_pyro_params']
                if name in _pyro_params:
                    # Update value of an existing PyroParam.
                    constraint, event_dim = _pyro_params[name]
                    unconstrained_value = getattr(self, name + "_unconstrained")
                    with torch.no_grad():
                        unconstrained_value.data = transform_to(constraint).inv(value.detach())
                    return

        if isinstance(value, PyroSample):
            # Create a new PyroSample, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            _pyro_samples = self.__dict__['_pyro_samples']
            _pyro_samples[name] = value.prior
            return

        super().__setattr__(name, value)

    def __delattr__(self, name):
        if '_pyro_params' in self.__dict__:
            _pyro_params = self.__dict__['_pyro_params']
            if name in _pyro_params:
                delattr(self, name + "_unconstrained")
                del _pyro_params[name]
                return

        if '_pyro_samples' in self.__dict__:
            _pyro_samples = self.__dict__['_pyro_samples']
            if name in _pyro_samples:
                del _pyro_samples[name]
                return

        super().__delattr__(name)
