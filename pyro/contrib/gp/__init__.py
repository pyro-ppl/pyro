from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.distributions import transform_to

import pyro
import pyro.distributions as dist


class Parameterized(nn.Module):
    """
    Parameterized class.

    This is a base class for other classes in Gaussian Process.
    By default, a parameter will be a :class:`torch.nn.Parameter` containing :class:`torch.FloatTensor`.
    To cast them to the correct data type or GPU device, we can call methods such as
    ``.double()``, ``.cuda(device=0)``,...
    See :class:`torch.nn.Module` for more information.

    :param str name: Name of this module.
    """

    def __init__(self, name=None):
        super(Parameterized, self).__init__()
        self._priors = {}
        self._constraints = {}
        self._fixed_params = {}
        self._registered_params = {}

        self.name = name

    def set_prior(self, param, prior):
        """
        Sets a prior to a parameter.

        :param str param: Name of a parameter.
        :param pyro.distributions.distribution.Distribution prior: A prior
            distribution for random variable ``param``.
        """
        self._priors[param] = prior

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of a parameter.
        :param torch.distributions.constraints.Constraint constraint: A Pytorch constraint.
            See :mod:`torch.distributions.constraints` for a list of constraints.
        """
        self._constraints[param] = constraint

    def fix_param(self, param, value=None):
        """
        Fixes a parameter to a specic value. If ``value=None``, fixes the parameter to the
        default value.

        :param str param: Name of a parameter.
        :param torch.Tensor value: A tensor to be fixed to ``param``.
        """
        if value is None:
            value = getattr(self, param).detach()
        self._fixed_params[param] = value

    def set_mode(self, mode):
        """
        Sets ``mode`` for the module to be able to use its parameters in stochastic functions.
        It also sets ``mode`` for submodules which belong to :class:`Parameterized` class.

        :param str mode: Either "model" or "guide".
        """
        if mode not in ["model", "guide"]:
            raise ValueError("Mode should be either 'model' or 'guide', but got {}.".format(mode))
        for module in self.children():
            if isinstance(module, Parameterized):
                module.set_mode(mode)
        for param in self._parameters:
            self._register_param(param, mode)

    def get_param(self, param):
        """
        Gets variable to be used in stochastic functions. The correct behavior will depend on
        the current ``mode`` of the module.

        :param str param: Name of a parameter.
        """
        if param not in self._registered_params:  # set_mode() has not been called yet
            return getattr(self, param)
        else:
            return self._registered_params[param]

    def _register_param(self, param, mode="model"):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for ``pyro.param()`` and
        ``pyro.sample()`` calls.

        :param str param: Name of a parameter.
        :param str mode: Either "model" or "guide".
        """
        if param in self._fixed_params:
            self._registered_params[param] = self._fixed_params[param]
            return
        prior = self._priors.get(param)
        if self.name is None:
            param_name = param
        else:
            param_name = pyro.param_with_module_name(self.name, param)

        if prior is None:
            constraint = self._constraints.get(param)
            default_value = getattr(self, param)
            if constraint is None:
                p = pyro.param(param_name, default_value)
            else:
                # TODO: use `constraint_to` inside `pyro.param(...)` when available
                unconstrained_param_name = param_name + "_unconstrained"
                unconstrained_param_0 = torch.tensor(
                    transform_to(constraint).inv(default_value).data.clone(),
                    requires_grad=True)
                p = transform_to(constraint)(pyro.param(unconstrained_param_name,
                                                        unconstrained_param_0))
        elif mode == "model":
            p = pyro.sample(param_name, prior)
        else:  # prior != None and mode = "guide"
            MAP_param_name = param_name + "_MAP"
            MAP_param_0 = torch.tensor(prior.analytic_mean().data.clone(), requires_grad=True)
            MAP_param = pyro.param(MAP_param_name, MAP_param_0)
            p = pyro.sample(param_name, dist.Delta(MAP_param))

        self._registered_params[param] = p
