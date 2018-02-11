from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
import torch.nn as nn
from torch.distributions import transform_to

import pyro
import pyro.distributions as dist


class Parameterized(nn.Module):
    """
    Parameterized class.

    This is a base class for other classes in Gaussian Process.
    By default, a parameter will be a ``torch.nn.Parameter`` containing ``torch.FloatTensor``.
    To cast them to the correct data type or GPU device, we can call methods such as
    ``.double()``, ``.cuda(device=0)``,...
    See `torch.nn.Module
    <http://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ for more information.
    """

    def __init__(self):
        super(Parameterized, self).__init__()
        self._priors = {}
        self._constraints = {}
        self._name = None
        self._registered_params = {}

    def set_prior(self, param, prior):
        """
        Sets a prior to a parameter.

        :param str param: Name of a parameter.
        :param pyro.distributions.Distribution prior: A prior distribution for random variable ``param``.
        """
        self._priors[param] = prior

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of a parameter.
        :param torch.distributions.constraints.Constraint constraint: A Pytorch constraint.
            See `Pytorch's docs
            <http://pytorch.org/docs/master/distributions.html#module-torch.distributions.constraints>`_
            for a list of constraints.
        """
        self._constraints[param] = constraint

    def set_mode(self, mode):
        """
        Sets mode for the module to be able to use its parameters in stochastic functions.

        :param str mode: Either "model" or "guide".
        """
        if mode not in ["model", "guide"]:
            raise ValueError("Mode should be either 'model' or 'guide', but got {}.".format(mode))
        for param in self._parameters:
            self._register_param(param, mode)
        return self

    def get_param(self, param):
        """
        Gets param to be used in stochastic functions. The correct behavior will depend on
        the current `mode` of the module.

        :param str param: Name of a parameter.
        """
        if param not in self._registered_params:  # set_mode() has not been called yet
            return getattr(self, param)
        else:
            return self._registered_params[param]

    def _register_param(self, param, mode="model"):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for `pyro.param()` and
        `pyro.sample()` calls.

        :param str param: Name of a parameter.
        :param str mode: Either "model" or "guide".
        """
        prior = self._priors[param] if param in self._priors else None
        if self._name is None:
            param_name = param
        else:
            param_name = pyro.param_with_module_name(self._name, param)

        if prior is None:
            constraint = self._constraints[param] if param in self._constraints else None
            tmp = getattr(self, param)
            if constraint is None:
                p = pyro.param(param_name, tmp)
            else:
                # TODO: use `constraint_to` inside `pyro.param(...)` when available
                unconstrained_param_name = param_name + "_unconstrained"
                unconstrained_param_0 = Variable(transform_to(constraint).inv(tmp).data.clone(),
                                                 requires_grad=True)
                p = transform_to(constraint)(pyro.param(unconstrained_param_name,
                                                        unconstrained_param_0))
        elif mode == "model":
            p = pyro.sample(param_name, prior)
        else:  # prior != None and mode = "guide"
            MAP_param_name = param_name + "_MAP"
            MAP_param_0 = Variable(prior.torch_dist.mean.data.clone(), requires_grad=True)
            MAP_param = pyro.param(MAP_param_name, MAP_param_0)
            p = pyro.sample(param_name, dist.Delta(MAP_param))

        self._registered_params[param] = p
