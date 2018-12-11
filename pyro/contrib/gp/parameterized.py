from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch.nn as nn
from torch.distributions import biject_to, constraints, transform_to
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib import autoname
from pyro.distributions.util import eye_like


def _get_independent_support(dist_instance):
    # XXX Should we treat the case dist_instance is Independent(Independent(Normal))?
    if isinstance(dist_instance, dist.Independent):
        return dist_instance.base_dist.support
    else:
        return dist_instance.support


class Parameterized(nn.Module):
    """
    A wrapper of :class:`torch.nn.Module` whose parameters can be set
    constraints, set priors.

    Under the hood, we move parameters to a buffer store and create "root"
    parameters which are used to generate that parameter's value. For example,
    if we set a contraint to a parameter, an "unconstrained" parameter will be
    created, and the constrained value will be transformed from that
    "unconstrained" parameter.

    By default, when we set a prior to a parameter, an auto Delta guide will be
    created. We can use the method :meth:`autoguide` to setup other auto guides.
    To fix a parameter to a specific value, it is enough to turn off its "root"
    parameters' ``requires_grad`` flags.

    Example::

        >>> class Linear(Parameterized):
        ...     def __init__(self, a, b):
        ...         super(Linear, self).__init__()
        ...         self.a = Parameter(a)
        ...         self.b = Parameter(b)
        ...
        ...     def forward(self, x):
        ...         return self.a * x + self.b
        ...
        >>> linear = Linear(torch.tensor(1.), torch.tensor(0.))
        >>> linear.set_constraint("a", constraints.positive)
        >>> linear.set_prior("b", dist.Normal(0, 1))
        >>> linear.autoguide("b", dist.Normal)
        >>> assert "a_unconstrained" in dict(linear.named_parameters())
        >>> assert "b_loc" in dict(linear.named_parameters())
        >>> assert "b_scale_unconstrained" in dict(linear.named_parameters())
        >>> assert "a" in dict(linear.named_buffers())
        >>> assert "b" in dict(linear.named_buffers())
        >>> assert "b_scale" in dict(linear.named_buffers())

    Note that by default, data of a parameter is a float :class:`torch.Tensor`
    (unless we use :func:`torch.set_default_tensor_type` to change default
    tensor type). To cast these parameters to a correct data type or GPU device,
    we can call methods such as :meth:`~torch.nn.Module.double` or
    :meth:`~torch.nn.Module.cuda`. See :class:`torch.nn.Module` for more
    information.
    """
    def __init__(self):
        super(Parameterized, self).__init__()
        self._constraints = OrderedDict()
        self._priors = OrderedDict()
        self._guides = OrderedDict()
        self._mode = None

    def set_constraint(self, name, constraint):
        """
        Sets the constraint of an existing parameter.

        :param str name: Name of the parameter.
        :param ~constraints.Constraint constraint: A PyTorch constraint. See
            :mod:`torch.distributions.constraints` for a list of constraints.
        """
        if constraint in [constraints.real, constraints.real_vector]:
            if name in self._constraints:  # delete previous constraints
                self._constraints.pop(name, None)
                self._parameters.pop("{}_unconstrained".format(name))

                if name not in self._priors:
                    # no prior -> no guide
                    # so we can move param back from buffer
                    p = Parameter(self._buffers.pop(name).detach())
                    self.register_parameter(name, p)
            return

        if name in self._priors:
            raise ValueError("Parameter {} already has a prior. Can not set a constraint for it."
                             .format(name))

        if name in self._parameters:
            p = self._parameters.pop(name)
        elif name in self._buffers:
            p = self._buffers[name]
        else:
            raise ValueError("There is no parameter with name: {}".format(name))

        p_unconstrained = Parameter(transform_to(constraint).inv(p).detach())
        self.register_parameter("{}_unconstrained".format(name), p_unconstrained)
        # due to precision issue, we might get f(f^-1(x)) != x
        # so it is necessary to transform back
        p = transform_to(constraint)(p_unconstrained)
        self.register_buffer(name, p.detach())
        self._constraints[name] = constraint

    def set_prior(self, name, prior):
        """
        Sets the constraint of an existing parameter.

        :param str name: Name of the parameter.
        :param ~pyro.distributions.distribution.Distribution prior: A Pyro prior
            distribution.
        """
        if name in self._parameters:
            # move param to _buffers
            p = self._parameters.pop(name)
            self.register_buffer(name, p)
        elif name not in self._buffers:
            raise ValueError("There is no parameter with name: {}".format(name))

        self._priors[name] = prior
        # remove the constraint and its unconstrained parameter
        self.set_constraint(name, constraints.real)

        self.autoguide(name, dist.Delta)

    def autoguide(self, name, dist_constructor):
        """
        Sets an autoguide for an existing parameter with name ``name`` (mimic
        the behavior of module :mod:`pyro.contrib.autoguide`).

        .. note:: `dist_constructor` should be one of
            :class:`~pyro.distributions.Delta`,
            :class:`~pyro.distributions.Normal`, and
            :class:`~pyro.distributions.MultivariateNormal`. More distribution
            constructor will be supported in the future if needed.

        :param str name: Name of the parameter.
        :param dist_constructor: A
            :class:`~pyro.distributions.distribution.Distribution` constructor.
        """
        if name not in self._priors:
            raise ValueError("There is no prior for parameter: {}".format(name))

        if dist_constructor not in [dist.Delta, dist.Normal, dist.MultivariateNormal]:
            raise NotImplementedError("Unsupported distribution type: {}"
                                      .format(dist_constructor))

        if name in self._guides:
            # delete previous guide's parameters
            dist_args = self._guides[name][1]
            for arg in dist_args:
                arg_name = "{}_{}".format(name, arg)
                if arg_name in self._constraints:
                    # delete its unconstrained parameter
                    self.set_constraint(arg_name, constraints.real)
                delattr(self, arg_name)

        # TODO: create a new argument `autoguide_args` to store other args for other
        # constructors. For example, in LowRankMVN, we need argument `rank`.
        p = self._buffers[name]
        if dist_constructor is dist.Delta:
            p_map = Parameter(p.detach())
            self.register_parameter("{}_map".format(name), p_map)
            self.set_constraint("{}_map".format(name), _get_independent_support(self._priors[name]))
            dist_args = {"map"}
        elif dist_constructor is dist.Normal:
            loc = Parameter(biject_to(self._priors[name].support).inv(p).detach())
            scale = Parameter(loc.new_ones(loc.shape))
            self.register_parameter("{}_loc".format(name), loc)
            self.register_parameter("{}_scale".format(name), scale)
            dist_args = {"loc", "scale"}
        elif dist_constructor is dist.MultivariateNormal:
            loc = Parameter(biject_to(self._priors[name].support).inv(p).detach())
            identity = eye_like(loc, loc.size(-1))
            scale_tril = Parameter(identity.repeat(loc.shape[:-1] + (1, 1)))
            self.register_parameter("{}_loc".format(name), loc)
            self.register_parameter("{}_scale_tril".format(name), scale_tril)
            dist_args = {"loc", "scale_tril"}
        else:
            raise NotImplementedError

        if dist_constructor is not dist.Delta:
            # each arg has a constraint, so we set constraints for them
            for arg in dist_args:
                self.set_constraint("{}_{}".format(name, arg),
                                    dist_constructor.arg_constraints[arg])
        self._guides[name] = (dist_constructor, dist_args)

    def set_mode(self, mode):
        """
        Sets ``mode`` of this object to be able to use its parameters in
        stochastic functions. If ``mode="model"``, a parameter will get its
        value from its prior. If ``mode="guide"``, the value will be drawn from
        its guide.

        .. note:: This method automatically sets ``mode`` for submodules which
            belong to :class:`Parameterized` class.

        :param str mode: Either "model" or "guide".
        """
        with autoname.name_count():
            for module in self.modules():
                if isinstance(module, Parameterized):
                    module.mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
        # We should get buffer values for constrained params first
        # otherwise, autoguide will use the old buffer for `scale` or `scale_tril`
        for name in self._constraints:
            if name not in self._priors:
                self._register_param(name)
        for name in self._priors:
            self._register_param(name)

    def _sample_from_guide(self, name):
        dist_constructor, dist_args = self._guides[name]

        if dist_constructor is dist.Delta:
            p_map = getattr(self, "{}_map".format(name))
            return pyro.sample(name, dist.Delta(p_map, event_dim=p_map.dim()))

        # create guide
        dist_args = {arg: getattr(self, "{}_{}".format(name, arg)) for arg in dist_args}
        guide = dist_constructor(**dist_args)

        # no need to do transforms when support is real (for mean field ELBO)
        if _get_independent_support(self._priors[name]) is constraints.real:
            return pyro.sample(name, guide.to_event())

        # otherwise, we do inference in unconstrained space and transform the value
        # back to original space
        # TODO: move this logic to contrib.autoguide or somewhere else
        unconstrained_value = pyro.sample("{}_latent".format(name), guide.to_event(),
                                          infer={"is_auxiliary": True})
        transform = biject_to(self._priors[name].support)
        value = transform(unconstrained_value)
        log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
        return pyro.sample(name, dist.Delta(value, log_density.sum(), event_dim=value.dim()))

    def _register_param(self, name):
        """
        In "model" mode, lifts the parameter with name ``name`` to a random
        sample using a predefined prior (from :meth:`set_prior` method). In
        "guide" mode, we use the guide generated from :meth:`autoguide`.

        :param str name: Name of the parameter.
        """
        if name in self._priors:
            with autoname.scope(prefix=self._get_name()):
                if self.mode == "model":
                    p = pyro.sample(name, self._priors[name])
                else:
                    p = self._sample_from_guide(name)
        elif name in self._constraints:
            p_unconstrained = self._parameters["{}_unconstrained".format(name)]
            p = transform_to(self._constraints[name])(p_unconstrained)
        self.register_buffer(name, p)
