from __future__ import absolute_import, division, print_function

import warnings

import torch
import torch.nn as nn


ParamProperty = namedtuple("ParamProperty", ["prior,constraint,guide,fixed"])


class Parameterized(nn.Module):
    """
    Parameters of this object can be set priors, set constraints, or fixed to a
    specific value.

    By default, data of a parameter is a float :class:`torch.Tensor` (unless we use
    :func:`torch.set_default_tensor_type` to change default tensor type). To cast these
    parameters to a correct data type or GPU device, we can call methods such as
    :meth:`~torch.nn.Module.double` or :meth:`~torch.nn.Module.cuda`. See
    :class:`torch.nn.Module` for more information.

    :param str name: Name of this object.
    """
    def __init__(self, name=None):
        super(Parameterized, self).__init__()
        self.name = name
        self._priors = {}
        self._constraints = {}
        self._fixed_params = {}
        self._registered_params = {}
        

    def set_prior(self, param, prior):
        """
        Sets a prior to a parameter.

        :param str param: Name of the parameter.
        :param ~pyro.distributions.distribution.Distribution prior: A Pyro prior
            distribution.
        """
        self._priors[param] = prior

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of the parameter.
        :param ~torch.distributions.constraints.Constraint constraint: A PyTorch
            constraint. See :mod:`torch.distributions.constraints` for a list of
            constraints.
        """
        self._constraints[param] = constraint

    def fix_param(self, param, value=None):
        """
        Fixes a parameter to a specic value. If ``value=None``, fixes the parameter
        to the default value.

        :param str param: Name of the parameter.
        :param torch.Tensor value: Fixed value.
        """
        if value is None:
            value = getattr(self, param).detach()
        self._fixed_params[param] = value

    def set_mode(self, mode, recursive=True):
        """
        Sets ``mode`` of this object to be able to use its parameters in stochastic
        functions. If ``mode="model"``, a parameter with prior will get its value
        from the primitive :func:`pyro.sample`. If ``mode="guide"`` or there is no
        prior on a parameter, :func:`pyro.param` will be called.

        This method automatically sets ``mode`` for submodules which belong to
        :class:`Parameterized` class unless ``recursive=False``.

        :param str mode: Either "model" or "guide".
        :param bool recursive: A flag to tell if we want to set mode for all
            submodules.
        """
        if mode not in ["model", "guide"]:
            raise ValueError("Mode should be either 'model' or 'guide', but got {}."
                             .format(mode))
        if recursive:
            for module in self.children():
                if isinstance(module, Parameterized):
                    module.set_mode(mode)
        for param in self._parameters:
            self._register_param(param, mode)

    def get_param(self, param):
        """
        Gets the current value of a parameter. The correct behavior will depend on
        ``mode`` of this object (see :meth:`set_mode` method).

        :param str param: Name of the parameter.
        """
        if param not in self._registered_params:  # set_mode() has not been called yet
            return getattr(self, param)
        else:
            return self._registered_params[param]

    def _register_param(self, param, mode="model"):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for
        :func:`pyro.param` and :func:`pyro.sample` primitives.

        :param str param: Name of the parameter.
        :param str mode: Either "model" or "guide".
        """
        if param in self._fixed_params:
            self._registered_params[param] = self._fixed_params[param]
            return
        prior = self._priors.get(param)
        if self.name is None:
            param_name = param
        else:
            param_name = param_with_module_name(self.name, param)

        if prior is None:
            constraint = self._constraints.get(param)
            default_value = getattr(self, param)
            if constraint is None:
                p = pyro.param(param_name, default_value)
            else:
                p = pyro.param(param_name, default_value, constraint=constraint)
        elif mode == "model":
            p = pyro.sample(param_name, prior)
        else:  # prior != None and mode = "guide"
            MAP_param_name = param_name + "_MAP"
            # initiate randomly from prior
            MAP_param = pyro.param(MAP_param_name, prior)
            p = pyro.sample(param_name, dist.Delta(MAP_param))

        self._registered_params[param] = p

    def __getattr__(self, name):
        else
        super(Parameterized, self).__getattr__(name)