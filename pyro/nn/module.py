"""
Pyro includes a special class :class:`~pyro.nn.module.PyroModule` whose
attributes can be modified by Pyro effects. To create one of these attributes,
use either the :class:`PyroParam` struct or the :class:`PyroSample` struct::

    my_module = PyroModule()
    my_module.x = PyroParameter(torch.tensor(1.), constraint=constraints.positive)
    my_module.y = PyroSample(dist.Normal(0, 1))

"""
from collections import OrderedDict, namedtuple

import torch
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist

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


class PyroModule(torch.nn.Module):
    """
    Subclass of :class:`torch.nn.Module` that supports setting of
    :class:`PyroParam` and :class:`PyroSample`.
    """
    def __init__(self):
        self._pyro_name = ""
        self._pyro_params = OrderedDict()
        self._pyro_samples = OrderedDict()
        super().__init__()

    @property
    def pyro_name(self):
        return self._pyro_name

    @pyro_name.setter
    def pyro_name(self, name):
        self._pyro_name = name
        for key, value in self._modules.items():
            assert isinstance(value, PyroModule)
            value.pyro_name = _make_name(name, key)

    def __setattr__(self, name, value):
        if isinstance(value, PyroModule):
            try:
                delattr(self, name)
            except AttributeError:
                pass
            value.pyro_name = _make_name(self._pyro_name, name)
            super().__setattr__(name, value)
            return

        if isinstance(value, PyroParam):
            try:
                delattr(self, name)
            except AttributeError:
                pass
            _pyro_params = self.__dict__['_pyro_params']
            constrained_value, constraint, event_dim = value
            _pyro_params[name] = constraint, event_dim
            with torch.no_grad():
                constrained_value = constrained_value.detach()
                unconstrained_value = transform_to(constraint).inv(constrained_value)
                unconstrained_value = unconstrained_value.contiguous()
            unconstrained_value = torch.nn.Parameter(unconstrained_value)
            unconstrained_name = name + "_unconstrained"
            setattr(self, unconstrained_name, unconstrained_value)
            return

        if isinstance(value, PyroSample):
            try:
                delattr(self, name)
            except AttributeError:
                pass
            _pyro_samples = self.__dict__['_pyro_samples']
            _pyro_samples[name] = value.prior
            return

        if isinstance(value, torch.Tensor):
            if '_pyro_params' in self.__dict__:
                _pyro_params = self.__dict__['_pyro_params']
                if name in _pyro_params:
                    constraint, event_dim = _pyro_params[name]
                    unconstrained_value = getattr(self, name + "_unconstrained")
                    with torch.no_grad():
                        unconstrained_value.data = transform_to(constraint).inv(value.detach())
                    return

        super().__setattr__(name, value)

    def __getattr__(self, name):
        if '_pyro_params' in self.__dict__:
            _pyro_params = self.__dict__['_pyro_params']
            if name in _pyro_params:
                constraint, event_dim = _pyro_params[name]
                unconstrained_value = pyro.param(
                    _make_name(self._pyro_name, name + "_unconstrained"),
                    super().__getattr__(name + "_unconstrained"),
                    event_dim=event_dim)
                value = transform_to(constraint)(unconstrained_value)
                return value

        if '_pyro_samples' in self.__dict__:
            _pyro_samples = self.__dict__['_pyro_samples']
            if name in _pyro_samples:
                prior = _pyro_samples[name]
                if not isinstance(prior, (dist.Distribution, torch.distributions.Distribution)):
                    prior = prior(self)
                return pyro.sample(_make_name(self._pyro_name, name), prior)

        result = super().__getattr__(name)

        if isinstance(result, torch.nn.Module) and not isinstance(result, PyroModule):
            pyro.module(_make_name(self._pyro_name, name), result)

        return result

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
