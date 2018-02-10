from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
from torch.distributions import transform_to

import pyro
import pyro.distributions as dist


# TODO: remove this class
class InducingPoints(nn.Module):

    def __init__(self, Xu, name="inducing_points"):
        super(InducingPoints, self).__init__()
        self.inducing_points = Parameter(Xu)
        self.name = name

    def forward(self):
        return self.inducing_points


class Parameterized(nn.Module):
    """
    Parameterized module.

    This is a base class for other classes in this Gaussian Process module.
    The method `set_prior(param, prior)` will be used to set priors to parameters.
    The method `set_constraint(param, prior)` will be used to set constraints to parameters.
    We use `set_mode(mode)` to interchange between "model" and guide" (default).
    If a parameter has a prior, we will use Maximum A Posteriori (MAP) for its guide.
    The method `link_param(param)` is used as a wrapper for `pyro.param()` or
    `pyro.sample()` call.
    """

    def __init__(self):
        super(Parameterized, self).__init__()
        self._priors = {}
        self._constraints = {}
        self._mode = "guide"
        self._name = None

    def set_prior(self, param, prior):
        self._priors[param] = prior

    def set_constraint(self, param, constraint):
        self._constraints[param] = constraint

    def set_mode(self, mode):
        if mode not in ["model", "guide"]:
            raise ValueError("Mode should be either 'model' or 'guide', but got {}.".format(mode))
        self._mode = mode

    def link_param(self, param):
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
        elif self._mode == "model":
            p = pyro.sample(param_name, prior)
        else:  # prior != None and mode = "guide"
            MAP_param_name = param_name + "_MAP"
            MAP_param_0 = Variable(prior.torch_dist.mean.data.clone(), requires_grad=True)
            MAP_param = pyro.param(MAP_param_name, MAP_param_0)
            p = pyro.sample(param_name, dist.Delta(MAP_param))

        return p
