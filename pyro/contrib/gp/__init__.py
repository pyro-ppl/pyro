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
    """

    def __init__(self):
        super(Parameterized, self).__init__()
        self._priors = {}
        self._constraints = {}
        self._mode = "guide"
        self._name = None

    def set_prior(self, param, prior):
        """
        Sets a prior to a parameter.

        :param str param: Name of a parameter.
        :param pyro.distributions.Distribution prior: A prior distribution for random variable `param`.
        """
        self._priors[param] = prior

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of a parameter.
        :param torch.distributions.constraints.Constraint constraint: A pytorch constraint.
            See `Pytorch's docs
            <http://pytorch.org/docs/master/distributions.html#module-torch.distributions.constraints>`_
            for a list of constraints.
        """
        self._constraints[param] = constraint

    def set_mode(self, mode):
        """
        Sets mode for the module. `self.link_param(param)` method will used this mode to
        decide its logic.

        :param str mode: Either "model" or "guide".
        """
        if mode not in ["model", "guide"]:
            raise ValueError("Mode should be either 'model' or 'guide', but got {}.".format(mode))
        self._mode = mode

    def register_param(self, param):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for `pyro.param()` and
        `pyro.sample()` calls.

        :param str param: Name of a parameter.
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
        elif self._mode == "model":
            p = pyro.sample(param_name, prior)
        else:  # prior != None and mode = "guide"
            MAP_param_name = param_name + "_MAP"
            MAP_param_0 = Variable(prior.torch_dist.mean.data.clone(), requires_grad=True)
            MAP_param = pyro.param(MAP_param_name, MAP_param_0)
            p = pyro.sample(param_name, dist.Delta(MAP_param))

        return p
