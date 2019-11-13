from collections import OrderedDict
from functools import partial

import torch
from torch.distributions import biject_to, constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.util import eye_like
from pyro.nn.module import PyroModule, PyroParam, PyroSample, _make_name


def _get_independent_support(dist_instance):
    if isinstance(dist_instance, dist.Independent):
        return _get_independent_support(dist_instance.base_dist)
    else:
        return dist_instance.support


class _Mode:
    """
    Sometimes-active cache for ``Parameterized.set_mode(mode)`` contexts.
    """
    def __init__(self, fn, mode):
        self.fn = fn
        self.current_mode = fn.mode
        for m in self.fn.modules():
            if isinstance(m, Parameterized):
                m.mode = mode

    def __enter__(self):
        self.fn._pyro_cache.__enter__()
        for m in self.fn.modules():
            if isinstance(m, Parameterized):
                for name in m._pyro_samples:
                    getattr(m, name)

    def __exit__(self, type, value, traceback):
        self.fn._pyro_cache.__exit__(type, value, traceback)
        for m in self.fn.modules():
            if isinstance(m, Parameterized):
                m.mode = self.current_mode


class Parameterized(PyroModule):
    """
    A wrapper of :class:`~pyro.nn.module.PyroModule` whose parameters can be set
    constraints, set priors.

    By default, when we set a prior to a parameter, an auto Delta guide will be
    created. We can use the method :meth:`autoguide` to setup other auto guides.

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
        >>> linear.a = PyroParam(torch.tensor(1.), constraints.positive)
        >>> linear.b = PyroSample(dist.Normal(0, 1))
        >>> linear.autoguide("b", dist.Normal)
        >>> assert "a_unconstrained" in dict(linear.named_parameters())
        >>> assert "b_loc" in dict(linear.named_parameters())
        >>> assert "b_scale_unconstrained" in dict(linear.named_parameters())

    Note that by default, data of a parameter is a float :class:`torch.Tensor`
    (unless we use :func:`torch.set_default_tensor_type` to change default
    tensor type). To cast these parameters to a correct data type or GPU device,
    we can call methods such as :meth:`~torch.nn.Module.double` or
    :meth:`~torch.nn.Module.cuda`. See :class:`torch.nn.Module` for more
    information.
    """
    def __init__(self):
        super(Parameterized, self).__init__()
        self._priors = OrderedDict()
        self._guides = OrderedDict()
        self._mode = "model"
        self.set_mode = partial(_Mode, self)

    def __setattr__(self, name, value):
        if isinstance(value, PyroSample):
            prior = value.prior
            if isinstance(prior, (dist.Distribution, torch.distributions.Distribution)):
                self._priors[name] = prior
                self.autoguide(name, dist.Delta)
                value = PyroSample(lambda self: prior if self.mode == "model" else self._guides[name])
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._priors:
            del self._priors[name]
            del self._guides[name]

        super().__delattr__(name)

    def autoguide(self, name, dist_constructor):
        """
        Sets an autoguide for an existing parameter with name ``name`` (mimic
        the behavior of module :mod:`pyro.infer.autoguide`).

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

        # delete old guide
        if name in self._guides:
            dist_args = self._guides[name][1]
            for arg in dist_args:
                delattr(self, "{}_{}".format(name, arg))

        p = self._priors[name]()  # init_to_sample strategy
        if dist_constructor is dist.Delta:
            support = _get_independent_support(self._priors[name])
            if support is constraints.real:
                p_map = Parameter(p.detach())
            else:
                p_map = PyroParam(p.detach(), support)
            setattr(self, "{}_map".format(name), p_map)
            dist_args = ("map",)
        elif dist_constructor is dist.Normal:
            loc = Parameter(biject_to(self._priors[name].support).inv(p).detach())
            scale = PyroParam(loc.new_ones(loc.shape), constraints.positive)
            setattr(self, "{}_loc".format(name), loc)
            setattr(self, "{}_scale".format(name), scale)
            dist_args = ("loc", "scale")
        elif dist_constructor is dist.MultivariateNormal:
            loc = Parameter(biject_to(self._priors[name].support).inv(p).detach())
            identity = eye_like(loc, loc.size(-1))
            scale_tril = PyroParam(identity.repeat(loc.shape[:-1] + (1, 1)),
                                   constraints.lower_cholesky)
            setattr(self, "{}_loc".format(name), loc)
            setattr(self, "{}_scale_tril".format(name), scale_tril)
            dist_args = ("loc", "scale_tril")
        else:
            raise NotImplementedError

        self._guides[name] = (dist_constructor, dist_args)

    def _get_guide(self, name, full_name):
        dist_constructor, dist_args = self._guides[name]

        if dist_constructor is dist.Delta:
            p_map = getattr(self, "{}_map".format(name))
            return dist.Delta(p_map, event_dim=p_map.dim())

        # create guide
        dist_args = {arg: getattr(self, "{}_{}".format(name, arg)) for arg in dist_args}
        guide = dist_constructor(**dist_args)

        # no need to do transforms when support is real (for mean field ELBO)
        if _get_independent_support(self._priors[name]) is constraints.real:
            return guide.to_event()

        # otherwise, we do inference in unconstrained space and transform the value
        # back to original space
        # TODO: move this logic to infer.autoguide or somewhere else
        unconstrained_value = pyro.sample("{}_latent".format(full_name), guide.to_event(),
                                          infer={"is_auxiliary": True})
        transform = biject_to(self._priors[name].support)
        value = transform(unconstrained_value)
        log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
        return dist.Delta(value, log_density.sum(), event_dim=value.dim())

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
