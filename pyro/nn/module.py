from collections import OrderedDict, namedtuple

import torch
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist

pyro_param = namedtuple("pyro_param", ("init_value", "constraint", "event_dim"))
pyro_param.__new__.__defaults__ = (constraints.real, None)

pyro_sample = namedtuple("pyro_sample", ("prior",))


class PyroModuleMixin:
    def __init__(self, name="model"):
        assert isinstance(name, str)
        self._pyro_name = name
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
            assert isinstance(value, PyroModuleMixin)
            value.pyro_name = "{}.{}".format(name, key)

    def __setattr__(self, name, value):
        if isinstance(value, torch.nn.Module):
            if not isinstance(value, PyroModuleMixin):
                raise TypeError
            value.pyro_name = "{}.{}".format(self._pyro_name, name)
            super().__setattr__(name, value)
            return

        if isinstance(value, pyro_param):
            _pyro_params = self.__dict__['_pyro_params']
            constrained_value, constraint, event_dim = value
            _pyro_params[name] = constraint, event_dim
            with torch.no_grad():
                if callable(constrained_value):
                    constrained_value = constrained_value()
                constrained_value = constrained_value.detach()
                unconstrained_value = transform_to(constraint).inv(constrained_value)
                unconstrained_value = unconstrained_value.contiguous()
            unconstrained_value = torch.nn.Parameter(unconstrained_value)
            unconstrained_name = name + "_unconstrained"
            setattr(self, unconstrained_name, unconstrained_value)
            return

        if isinstance(value, pyro_sample):
            _pyro_samples = self.__dict__['_pyro_samples']
            _pyro_samples[name] = value.prior
            return

        return super().__setattr__(name, value)

    def __getattr__(self, name):
        if '_pyro_params' in self.__dict__:
            _pyro_params = self.__dict__['_pyro_params']
            if name in _pyro_params:
                constraint, event_dim = _pyro_params[name]
                unconstrained_value = pyro.param(
                    "{}.{}_unconstrained".format(self._pyro_name, name),
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
                return pyro.sample("{}.{}".format(self._pyro_name, name), prior)

        result = super().__getattr__(name)

        if isinstance(result, torch.nn.Module) and not isinstance(result, PyroModuleMixin):
            pyro.module("{}.{}".format(self._pyro_name, name), result)

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


class PyroModule(PyroModuleMixin, torch.nn.Module):
    """
    Subclass of :class:`torch.nn.Module` that supports setting of
    :class:`pyro_param` and :class:`pyro_sample` attributes. To add this behavior
    to an existing module, use :class:`PyroModuleMixin` instead.
    """
    pass
