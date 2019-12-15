"""
Pyro includes an experimental class :class:`~pyro.nn.module.PyroModule`, a
subclass of :class:`torch.nn.Module`, whose attributes can be modified by Pyro
effects.  To create a poutine-aware attribute, use either the
:class:`PyroParam` struct or the :class:`PyroSample` struct::

    my_module = PyroModule()
    my_module.x = PyroParam(torch.tensor(1.), constraint=constraints.positive)
    my_module.y = PyroSample(dist.Normal(0, 1))

"""
import functools
from collections import OrderedDict, namedtuple

import torch
from torch.distributions import constraints, transform_to

import pyro
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


def _unconstrain(constrained_value, constraint):
    with torch.no_grad():
        unconstrained_value = transform_to(constraint).inv(constrained_value.detach())
        return torch.nn.Parameter(unconstrained_value)


class _Context:
    """
    Sometimes-active cache for ``PyroModule.__call__()`` contexts.
    """
    def __init__(self):
        self.active = 0
        self.cache = {}
        self.used = False

    def __enter__(self):
        self.active += 1
        self.used = True

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


def _get_pyro_params(module):
    for name in module._parameters:
        if name.endswith("_unconstrained"):
            constrained_name = name[:-len("_unconstrained")]
            if isinstance(module, PyroModule) and constrained_name in module._pyro_params:
                yield constrained_name, getattr(module, constrained_name)
                continue
        yield name, module._parameters[name]


class _PyroModuleMeta(type):
    _pyro_mixin_cache = {}

    # Unpickling helper to create an empty object of type PyroModule[Module].
    class _New:
        def __init__(self, Module):
            self.__class__ = PyroModule[Module]

    def __getitem__(cls, Module):
        assert isinstance(Module, type)
        assert issubclass(Module, torch.nn.Module)
        if issubclass(Module, PyroModule):
            return Module
        if Module is torch.nn.Module:
            return PyroModule
        if Module in _PyroModuleMeta._pyro_mixin_cache:
            return _PyroModuleMeta._pyro_mixin_cache[Module]

        class result(Module, PyroModule):
            # Unpickling helper to load an object of type PyroModule[Module].
            def __reduce__(self):
                state = getattr(self, '__getstate__', self.__dict__.copy)()
                return _PyroModuleMeta._New, (Module,), state

        result.__name__ = "Pyro" + Module.__name__
        _PyroModuleMeta._pyro_mixin_cache[Module] = result
        return result


class PyroModule(torch.nn.Module, metaclass=_PyroModuleMeta):
    """
    EXPERIMENTAL Subclass of :class:`torch.nn.Module` whose attributes can be
    modified by Pyro effects. Attributes can be set using helpers
    :class:`PyroParam` and :class:`PyroSample` , and methods can be decorated
    by :func:`pyro_method` .

    **Parameters**

    To create a Pyro-managed parameter attribute, set that attribute using
    either :class:`torch.nn.Parameter` (for unconstrained parameters) or
    :class:`PyroParam` (for constrained parameters). Reading that attribute
    will then trigger a :func:`pyro.param` statement. For example::

        # Create Pyro-managed parameter attributes.
        my_module = PyroModule()
        my_module.loc = nn.Parameter(torch.tensor(0.))
        my_module.scale = PyroParam(torch.tensor(1.),
                                    constraint=constraints.positive)
        # Read the attributes.
        loc = my_module.loc  # Triggers a pyro.param statement.
        scale = my_module.scale  # Triggers another pyro.param statement.

    Note that, unlike normal :class:`torch.nn.Module` s, :class:`PyroModule` s
    should not be registered with :func:`pyro.module` statements.
    :class:`PyroModule` s can contain other :class:`PyroModule` s and normal
    :class:`torch.nn.Module` s.  Accessing a normal :class:`torch.nn.Module`
    attribute of a :class:`PyroModule` triggers a :func:`pyro.module`
    statement.  If multiple :class:`PyroModule` s appear in a single Pyro model
    or guide, they should be included in a single root :class:`PyroModule` for
    that model.

    :class:`PyroModule` s synchronize data with the param store at each
    ``setattr``, ``getattr``, and ``delattr`` event, based on the nested name
    of an attribute:

    -   Setting ``mod.x = x_init`` tries to read ``x`` from the param store. If a
        value is found in the param store, that value is copied into ``mod``
        and ``x_init`` is ignored; otherwise ``x_init`` is copied into both
        ``mod`` and the param store.
    -   Reading ``mod.x`` tries to read ``x`` from the param store. If a
        value is found in the param store, that value is copied into ``mod``;
        otherwise ``mod``'s value is copied into the param store. Finally
        ``mod`` and the param store agree on a single value to return.
    -   Deleting ``del mod.x`` removes a value from both ``mod`` and the param
        store.

    Note two :class:`PyroModule` of the same name will both synchronize with
    the global param store and thus contain the same data.  When creating a
    :class:`PyroModule`, then deleting it, then creating another with the same
    name, the latter will be populated with the former's data from the param
    store. To avoid this persistence, either ``pyro.clear_param_store()`` or
    call :func:`clear` before deleting a :class:`PyroModule` .

    :class:`PyroModule` s can be saved and loaded either directly using
    :func:`torch.save` / :func:`torch.load` or indirectly using the param
    store's :meth:`~pyro.params.param_store.ParamStoreDict.save` /
    :meth:`~pyro.params.param_store.ParamStoreDict.load` . Note that
    :func:`torch.load` will be overridden by any values in the param store, so
    it is safest to ``pyro.clear_param_store()`` before loading.

    **Samples**

    To create a Pyro-managed random attribute, set that attribute using the
    :class:`PyroSample` helper, specifying a prior distribution. Reading that
    attribute will then trigger a :func:`pyro.sample` statement. For example::

        # Create Pyro-managed random attributes.
        my_module.x = PyroSample(dist.Normal(0, 1))
        my_module.y = PyroSample(lambda self: dist.Normal(self.loc, self.scale))

        # Sample the attributes.
        x = my_module.x  # Triggers a pyro.sample statement.
        y = my_module.y  # Triggers one pyro.sample + two pyro.param statements.

    Sampling is cached within each invocation of ``.__call__()`` or method
    decorated by :func:`pyro_method` .  Because sample statements can appear
    only once in a Pyro trace, you should ensure that traced access to sample
    attributes is wrapped in a single invocation of ``.__call__()`` or method
    decorated by :func:`pyro_method` .

    To make an existing module probabilistic, you can create a subclass and
    overwrite some parameters with :class:`PyroSample` s::

        class RandomLinear(nn.Linear, PyroModule):  # used as a mixin
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
                self.weight = PyroSample(
                    lambda self: dist.Normal(0, 1)
                                     .expand([self.out_features,
                                              self.in_features])
                                     .to_event(2))

    **Mixin classes**

    :class:`PyroModule` can be used as a mixin class, and supports simple
    syntax for dynamically creating mixins, for example the following are
    equivalent::

        # Version 1. create a named mixin class
        class PyroLinear(nn.Linear, PyroModule):
            pass

        m.linear = PyroLinear(m, n)

        # Version 2. create a dynamic mixin class
        m.linear = PyroModule[nn.Linear](m, n)

    This notation can be used recursively to create Bayesian modules, e.g.::

        model = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](28 * 28, 100),
            PyroModule[nn.Sigmoid](),
            PyroModule[nn.Linear](100, 100),
            PyroModule[nn.Sigmoid](),
            PyroModule[nn.Linear](100, 10),
        )
        assert isinstance(model, nn.Sequential)
        assert isinstance(model, PyroModule)

        # Now we can be Bayesian about weights in the first layer.
        model[0].weight = PyroSample(
            prior=dist.Normal(0, 1).expand([28 * 28, 100]).to_event(2))
        guide = AutoDiagonalNormal(model)

    Note that ``PyroModule[...]`` does not recursively mix in
    :class:`PyroModule` to submodules of the input ``Module``; hence we needed
    to wrap each submodule of the ``nn.Sequential`` above.

    :param str name: Optional name for a root PyroModule. This is ignored in
        sub-PyroModules of another PyroModule.
    """
    def __init__(self, name=""):
        self._pyro_name = name
        self._pyro_context = _Context()  # shared among sub-PyroModules
        self._pyro_params = OrderedDict()
        self._pyro_samples = OrderedDict()
        super().__init__()

    def add_module(self, name, module):
        """
        Adds a child module to the current module.
        """
        if isinstance(module, PyroModule):
            module._pyro_set_supermodule(_make_name(self._pyro_name, name), self._pyro_context)
        super().add_module(name, module)

    def named_pyro_params(self, prefix='', recurse=True):
        """
        Returns an iterator over PyroModule parameters, yielding both the
        name of the parameter as well as the parameter itself.

        :param str prefix: prefix to prepend to all parameter names.
        :param bool recurse: if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.
        :returns: a generator which yields tuples containing the name and parameter
        """
        gen = self._named_members(_get_pyro_params, prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def _pyro_set_supermodule(self, name, context):
        self._pyro_name = name
        self._pyro_context = context
        for key, value in self._modules.items():
            if isinstance(value, PyroModule):
                assert not value._pyro_context.used, \
                    "submodule {} has executed outside of supermodule".format(name)
                value._pyro_set_supermodule(_make_name(name, key), context)

    def _pyro_get_fullname(self, name):
        assert self.__dict__['_pyro_context'].used, "fullname is not yet defined"
        return _make_name(self.__dict__['_pyro_name'], name)

    def __call__(self, *args, **kwargs):
        with self._pyro_context:
            return super().__call__(*args, **kwargs)

    def __getattr__(self, name):
        # PyroParams trigger pyro.param statements.
        if '_pyro_params' in self.__dict__:
            _pyro_params = self.__dict__['_pyro_params']
            if name in _pyro_params:
                constraint, event_dim = _pyro_params[name]
                unconstrained_value = getattr(self, name + "_unconstrained")
                if self._pyro_context.active:
                    fullname = self._pyro_get_fullname(name)
                    if fullname in _PYRO_PARAM_STORE:
                        if _PYRO_PARAM_STORE._params[fullname] is not unconstrained_value:
                            # Update PyroModule <--- ParamStore.
                            unconstrained_value = _PYRO_PARAM_STORE._params[fullname]
                            if not isinstance(unconstrained_value, torch.nn.Parameter):
                                # Update PyroModule ---> ParamStore (type only; data is preserved).
                                unconstrained_value = torch.nn.Parameter(unconstrained_value)
                                _PYRO_PARAM_STORE._params[fullname] = unconstrained_value
                                _PYRO_PARAM_STORE._param_to_name[unconstrained_value] = fullname
                            super().__setattr__(name + "_unconstrained", unconstrained_value)
                    else:
                        # Update PyroModule ---> ParamStore.
                        _PYRO_PARAM_STORE._constraints[fullname] = constraint
                        _PYRO_PARAM_STORE._params[fullname] = unconstrained_value
                        _PYRO_PARAM_STORE._param_to_name[unconstrained_value] = fullname
                    return pyro.param(fullname, event_dim=event_dim)
                else:  # Cannot determine supermodule and hence cannot compute fullname.
                    return transform_to(constraint)(unconstrained_value)

        # PyroSample trigger pyro.sample statements.
        if '_pyro_samples' in self.__dict__:
            _pyro_samples = self.__dict__['_pyro_samples']
            if name in _pyro_samples:
                prior = _pyro_samples[name]
                context = self._pyro_context
                if context.active:
                    fullname = self._pyro_get_fullname(name)
                    value = context.get(fullname)
                    if value is None:
                        if not hasattr(prior, "sample"):  # if not a distribution
                            prior = prior(self)
                        value = pyro.sample(fullname, prior)
                        context.set(fullname, value)
                    return value
                else:  # Cannot determine supermodule and hence cannot compute fullname.
                    if not hasattr(prior, "sample"):  # if not a distribution
                        prior = prior(self)
                    return prior()

        result = super().__getattr__(name)

        # Regular nn.Parameters trigger pyro.param statements.
        if isinstance(result, torch.nn.Parameter) and not name.endswith("_unconstrained"):
            if self._pyro_context.active:
                pyro.param(self._pyro_get_fullname(name), result)

        # Regular nn.Modules trigger pyro.module statements.
        if isinstance(result, torch.nn.Module) and not isinstance(result, PyroModule):
            if self._pyro_context.active:
                pyro.module(self._pyro_get_fullname(name), result)

        return result

    def __setattr__(self, name, value):
        if isinstance(value, PyroModule):
            # Create a new sub PyroModule, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            self.add_module(name, value)
            return

        if isinstance(value, PyroParam):
            # Create a new PyroParam, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            constrained_value, constraint, event_dim = value
            self._pyro_params[name] = constraint, event_dim
            if self._pyro_context.active:
                fullname = self._pyro_get_fullname(name)
                if fullname in _PYRO_PARAM_STORE:
                    # Update PyroModule <--- ParamStore.
                    unconstrained_value = _PYRO_PARAM_STORE._params[fullname]
                    if not isinstance(unconstrained_value, torch.nn.Parameter):
                        # Update PyroModule ---> ParamStore (type only; data is preserved).
                        unconstrained_value = torch.nn.Parameter(unconstrained_value)
                        _PYRO_PARAM_STORE._params[fullname] = unconstrained_value
                        _PYRO_PARAM_STORE._param_to_name[unconstrained_value] = fullname
                else:
                    unconstrained_value = _unconstrain(constrained_value, constraint)
            else:  # Cannot determine supermodule and hence cannot compute fullname.
                unconstrained_value = _unconstrain(constrained_value, constraint)
            super().__setattr__(name + "_unconstrained", unconstrained_value)
            return

        if isinstance(value, torch.nn.Parameter):
            # Create a new nn.Parameter, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            if self._pyro_context.active:
                fullname = self._pyro_get_fullname(name)
                value = pyro.param(fullname, value)
                if not isinstance(value, torch.nn.Parameter):
                    # Update PyroModule ---> ParamStore (type only; data is preserved).
                    value = torch.nn.Parameter(value)
                    _PYRO_PARAM_STORE._params[fullname] = value
                    _PYRO_PARAM_STORE._param_to_name[value] = fullname
            super().__setattr__(name, value)
            return

        if isinstance(value, torch.Tensor):
            if name in self._pyro_params:
                # Update value of an existing PyroParam.
                constraint, event_dim = self._pyro_params[name]
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
        if name in self._parameters:
            del self._parameters[name]
            if self._pyro_context.used:
                fullname = self._pyro_get_fullname(name)
                if fullname in _PYRO_PARAM_STORE:
                    # Update PyroModule ---> ParamStore.
                    del _PYRO_PARAM_STORE[fullname]
            return

        if name in self._pyro_params:
            delattr(self, name + "_unconstrained")
            del self._pyro_params[name]
            if self._pyro_context.used:
                fullname = self._pyro_get_fullname(name)
                if fullname in _PYRO_PARAM_STORE:
                    # Update PyroModule ---> ParamStore.
                    del _PYRO_PARAM_STORE[fullname]
            return

        if name in self._pyro_samples:
            del self._pyro_samples[name]
            return

        if name in self._modules:
            del self._modules[name]
            if self._pyro_context.used:
                fullname = self._pyro_get_fullname(name)
                for p in list(_PYRO_PARAM_STORE.keys()):
                    if p.startswith(fullname):
                        del _PYRO_PARAM_STORE[p]
            return

        super().__delattr__(name)


def pyro_method(fn):
    """
    Decorator for top-level methods of a :class:`PyroModule` to enable pyro
    effects and cache ``pyro.sample`` statements.

    This should be applied to all public methods that read Pyro-managed
    attributes, but is not needed for ``.forward()``.
    """

    @functools.wraps(fn)
    def cached_fn(self, *args, **kwargs):
        with self._pyro_context:
            return fn(self, *args, **kwargs)

    return cached_fn


def clear(mod):
    """
    Removes data from both a :class:`PyroModule` and the param store.

    :param PyroModule mod: A module to clear.
    """
    assert isinstance(mod, PyroModule)
    for name in list(mod._pyro_params):
        delattr(mod, name)
    for name in list(mod._parameters):
        delattr(mod, name)
    for name in list(mod._modules):
        delattr(mod, name)


def to_pyro_module_(m, recurse=True):
    """
    Converts an ordinary :class:`torch.nn.Module` instance to a
    :class:`PyroModule` **in-place**.

    This is useful for adding Pyro effects to third-party modules: no
    third-party code needs to be modified. For example::

        model = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 10),
        )
        to_pyro_module_(model)
        assert isinstance(model, PyroModule[nn.Sequential])
        assert isinstance(model[0], PyroModule[nn.Linear])

        # Now we can attempt to be fully Bayesian:
        for m in model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(prior=dist.Normal(0, 1)
                                                      .expand(value.shape)
                                                      .to_event(value.dim())))
        guide = AutoDiagonalNormal(model)

    :param torch.nn.Module m: A module instance.
    :param bool recurse: Whether to convert submodules to :class:`PyroModules` .
    """
    if not isinstance(m, torch.nn.Module):
        raise TypeError("Expected an nn.Module instance but got a {}".format(type(m)))

    if isinstance(m, PyroModule):
        if recurse:
            for name, value in list(m._modules.items()):
                to_pyro_module_(value)
                setattr(m, name, value)
        return

    # Change m's type in-place.
    m.__class__ = PyroModule[m.__class__]
    m._pyro_name = ""
    m._pyro_context = _Context()
    m._pyro_params = OrderedDict()
    m._pyro_samples = OrderedDict()

    # Reregister parameters and submodules.
    for name, value in list(m._parameters.items()):
        setattr(m, name, value)
    for name, value in list(m._modules.items()):
        if recurse:
            to_pyro_module_(value)
        setattr(m, name, value)
