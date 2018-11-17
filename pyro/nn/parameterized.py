from __future__ import absolute_import, division, print_function

from collections import defaultdict

import torch.nn as nn
from torch.distributions import constraints 

import pyro
import pyro.distributions as dist


class Parameterized(nn.Module):
    """
    Base class for other modules in Gaussin Process module.

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
        self._constraints = defaultdict(constraints.real)
        self._priors = {}
        self._guides = defaultdict("Delta")
        self._registered_params = {}

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of the parameter.
        :param ~torch.distributions.constraints.Constraint constraint: A PyTorch
            constraint. See :mod:`torch.distributions.constraints` for a list of
            constraints.
        """
        self._constraints[param] = constraint

    def set_prior(self, param, prior):
        """
        Sets a prior to a parameter.

        :param str param: Name of the parameter.
        :param ~pyro.distributions.distribution.Distribution prior: A Pyro prior
            distribution.
        """
        self._priors[param] = prior

    def set_guide(self, param, guide):
        """
        Sets a prior to a parameter.

        :param str param: Name of the parameter.
        :param str guide: One of "Delta", "Normal", "MultivariateNormal".
        """
        self._guides[param] = guide

    def set_mode(self, mode):
        """
        Sets ``mode`` of this object to be able to use its parameters in stochastic
        functions. If ``mode="model"``, a parameter with prior will get its value
        from the primitive :func:`pyro.sample`. If ``mode="guide"`` or there is no
        prior on a parameter, :func:`pyro.param` will be called.

        This method automatically sets ``mode`` for submodules which belong to
        :class:`Parameterized` class.

        :param str mode: Either "prior" or "guide".
        """
        self.mode = mode
        for module in self.children():
            if isinstance(module, Parameterized):
                module.set_mode(mode)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if self._mode != mode:
            self._mode = mode
            self._register_param(param)

    def __setattr__(self, name, value):
        if name in self._registered_params:
            del self._registered_params[name]
        super(Parameterized, self).__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._registered_params:
            return self._registered_params[name]
        super(Parameterized, self).__getattr__(name)

    def _register_param(self, param):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for
        :func:`pyro.param` and :func:`pyro.sample` primitives.

        :param str param: Name of the parameter.
        """
        value = self._parameters[param]
        param_name = param_with_module_name(self.name, param) if self.name is not None else param
        if param in self._priors:
            if self.mode == "model":
                p = pyro.sample(param_name, self._priors[param])
            else:
                guide = self._guides[param]
                if guide == "Delta":
                    p_MAP = pyro.param("{}_MAP".format(param_name), self._priors[param])
                    p = pyro.sample(param_name, dist.Delta(p_MAP))
                elif guide == "Normal":
                    loc = pyro.param("{}_loc".format(param_name),
                                     lambda: value.new_zeros(value.shape))
                    scale = pyro.param("{}_scale".format(self.prefix),
                                       lambda: value.new_ones(value.shape),
                                       constraint=constraints.positive)
                    p = pyro.sample(param_name,
                        dist.Normal(loc, scale_tril=scale_tril).independent(value.dim()))
                elif guide == "MultivariateNormal":
                    n = value.size(-1)
                    loc = pyro.param("{}_loc".format(param_name),
                                       lambda: value.new_zeros(value.shape))
                    scale_tril = pyro.param("{}_scale_tril".format(param_name),
                        lambda: torch.eye(self.latent_dim, out=value.new_empty(n, n))
                            .repeat(value.shape[:-1] + (1, 1)),
                        constraint=constraints.lower_cholesky)
                    p = pyro.sample(param_name, dist.MultivariateNormal(p_loc,
                        scale_tril=p_scale_tril).independent(value.dim() - 1))
        else:
            p = pyro.param(param_name, value, constraint=self._constraints[param])
        self._registered_params[param] = p
