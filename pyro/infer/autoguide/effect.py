# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Union

import torch
from torch.distributions import biject_to, constraints

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.distribution import Distribution
from pyro.infer.effect_elbo import GuideMessenger
from pyro.nn.module import PyroModule, PyroParam, pyro_method
from pyro.poutine.runtime import get_plates

from .initialization import init_to_feasible, init_to_mean
from .utils import deep_getattr, deep_setattr, helpful_support_errors


class AutoMessengerMeta(type(GuideMessenger), type(PyroModule)):
    pass


class AutoMessenger(GuideMessenger, PyroModule, metaclass=AutoMessengerMeta):
    """
    EXPERIMENTAL Base class for :class:`pyro.infer.effect_elbo.GuideMessenger`
    autoguides.
    """

    def __call__(self, *args, **kwargs):
        self._outer_plates = get_plates()
        try:
            return super().__call__(*args, **kwargs)
        finally:
            del self._outer_plates

    def call(self, *args, **kwargs):
        """
        Method that calls :meth:`forward` and returns parameter values of the
        guide as a `tuple` instead of a `dict`, which is a requirement for
        JIT tracing. Unlike :meth:`forward`, this method can be traced by
        :func:`torch.jit.trace_module`.

        .. warning::
            This method may be removed once PyTorch JIT tracer starts accepting
            `dict` as valid return types. See
            `issue <https://github.com/pytorch/pytorch/issues/27743>_`.
        """
        result = self(*args, **kwargs)
        return tuple(v for _, v in sorted(result.items()))

    def _remove_outer_plates(self, value: torch.Tensor, event_dim: int) -> torch.Tensor:
        """
        Removes particle plates from initial values of parameters.
        """
        for f in self._outer_plates:
            dim = f.dim - event_dim
            if -value.dim() <= dim:
                dim = dim + value.dim()
                value = value[(slice(None),) * dim + (slice(1),)]
        for dim in range(value.dim() - event_dim):
            value = value.squeeze(0)
        return value


class AutoNormalMessenger(AutoMessenger):
    """
    EXPERIMENTAL Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` ,
    intended for use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or
    similar.

    The mean-field posterior at any site is a transformed normal distribution.

    Derived classes may override particular sites and use this simply as a
    default, e.g.::

        def model(data):
            a = pyro.sample("a", dist.Normal(0, 1))
            b = pyro.sample("b", dist.Normal(0, 1))
            c = pyro.sample("c", dist.Normal(a + b, 1))
            pyro.sample("obs", dist.Normal(c, 1), obs=data)

        class MyGuideMessenger(AutoNormalMessenger):
            def get_posterior(self, name, prior, upstream_values):
                if name == "c":
                    # Use a custom distribution at site c.
                    bias = pyro.param("c_bias", lambda: torch.zeros(()))
                    weight = pyro.param("c_weight", lambda: torch.ones(()),
                                        constraint=constraints.positive)
                    scale = pyro.param("c_scale", lambda: torch.ones(()),
                                       constraint=constraints.positive)
                    a = upstream_values["a"]
                    b = upstream_values["b"]
                    loc = bias + weight * (a + b)
                    return dist.Normal(loc, scale)
                # Fall back to mean field.
                return super().get_posterior(name, prior, upstream_values)

    Note that above we manually computed ``loc = bias + weight * (a + b)``.
    Alternatively we could reuse the model-side computation by setting ``loc =
    bias + weight * prior.loc``::

        class MyGuideMessenger_v2(AutoNormalMessenger):
            def get_posterior(self, name, prior, upstream_values):
                if name == "c":
                    # Use a custom distribution at site c.
                    bias = pyro.param("c_bias", lambda: torch.zeros(()))
                    scale = pyro.param("c_scale", lambda: torch.ones(()),
                                       constraint=constraints.positive)
                    weight = pyro.param("c_weight", lambda: torch.ones(()),
                                        constraint=constraints.positive)
                    loc = bias + weight * prior.loc
                    return dist.Normal(loc, scale)
                # Fall back to mean field.
                return super().get_posterior(name, prior, upstream_values)

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    """

    def __init__(
        self,
        model: Callable,
        *,
        init_loc_fn: Callable = init_to_mean(fallback=init_to_feasible),
        init_scale: float = 0.1,
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        super().__init__(model)
        self.init_loc_fn = init_loc_fn
        self._init_scale = init_scale
        self._computing_median = False

    @pyro_method
    def get_posterior(
        self,
        name: str,
        prior: Distribution,
        upstream_values: Dict[str, torch.Tensor],
    ) -> Union[Distribution, torch.Tensor]:
        if self._computing_median:
            return self._get_posterior_median(name, prior)

        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
        loc, scale = self._get_params(name, prior)
        posterior = dist.TransformedDistribution(
            dist.Normal(loc, scale).to_event(transform.domain.event_dim),
            transform.with_cache(),
        )
        return posterior

    def _get_params(self, name: str, prior: Distribution):
        try:
            loc = deep_getattr(self.locs, name)
            scale = deep_getattr(self.scales, name)
            return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with poutine.block(), torch.no_grad():
            transform = biject_to(prior.support)
            event_dim = transform.domain.event_dim
            constrained = self.init_loc_fn({"name": name, "fn": prior}).detach()
            unconstrained = transform.inv(constrained)
            init_loc = self._remove_outer_plates(unconstrained, event_dim)
            init_scale = torch.full_like(init_loc, self._init_scale)

        deep_setattr(self, "locs." + name, PyroParam(init_loc, event_dim=event_dim))
        deep_setattr(
            self,
            "scales." + name,
            PyroParam(init_scale, constraint=constraints.positive, event_dim=event_dim),
        )
        return self._get_params(name, prior)

    def median(self, *args, **kwargs):
        self._computing_median = True
        try:
            return self(*args, **kwargs)
        finally:
            self._computing_median = False

    def _get_posterior_median(self, name, prior):
        transform = biject_to(prior.support)
        loc, scale = self._get_params(name, prior)
        return transform(loc)


class AutoRegressiveMessenger(AutoMessenger):
    """
    EXPERIMENTAL Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` ,
    intended for use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or
    similar.

    The posterior at any site is a learned affine transform of the prior,
    conditioned on upstream posterior samples. The affine transform operates in
    unconstrained space. This supports only continuous latent variables.

    Derived classes may override particular sites and use this simply as a
    default, e.g.::

        class MyGuideMessenger(AutoRegressiveMessenger):
            def get_posterior(self, name, prior, upstream_values):
                if name == "x":
                    # Use a custom distribution at site x.
                    loc = pyro.param("x_loc", lambda: torch.zeros(prior.shape()))
                    scale = pyro.param("x_scale", lambda: torch.ones(prior.shape())),
                                       constraint=constraints.positive
                    return dist.Normal(loc, scale).to_event(prior.event_dim())
                # Fall back to autoregressive.
                return super().get_posterior(name, prior, upstream_values)

    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    """

    def __init__(
        self,
        model: Callable,
        *,
        init_loc_fn: Callable = init_to_mean(fallback=init_to_feasible),
        init_scale: float = 0.1,
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        super().__init__(model)
        self.init_loc_fn = init_loc_fn
        self._init_scale = init_scale

    @pyro_method
    def get_posterior(
        self,
        name: str,
        prior: Distribution,
        upstream_values: Dict[str, torch.Tensor],
    ) -> Union[Distribution, torch.Tensor]:
        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
        loc, scale = self._get_params(name, prior)
        affine = dist.transforms.AffineTransform(
            loc, scale, event_dim=transform.domain.event_dim, cache_size=1
        )
        posterior = dist.TransformedDistribution(
            prior, [transform.inv.with_cache(), affine, transform.with_cache()]
        )
        return posterior

    def _get_params(self, name: str, prior: Distribution):
        try:
            loc = deep_getattr(self.locs, name)
            scale = deep_getattr(self.scales, name)
            return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with poutine.block(), torch.no_grad():
            transform = biject_to(prior.support)
            event_dim = transform.domain.event_dim
            constrained = self.init_loc_fn({"name": name, "fn": prior}).detach()
            unconstrained = transform.inv(constrained)
            # Initialize the distribution to be an affine combination:
            #   init_scale * prior + (1 - init_scale) * init_loc
            init_loc = self._remove_outer_plates(unconstrained, event_dim)
            init_loc = init_loc * (1 - self._init_scale)
            init_scale = torch.full_like(init_loc, self._init_scale)

        deep_setattr(self, "locs." + name, PyroParam(init_loc, event_dim=event_dim))
        deep_setattr(
            self,
            "scales." + name,
            PyroParam(init_scale, constraint=constraints.positive, event_dim=event_dim),
        )
        return self._get_params(name, prior)
