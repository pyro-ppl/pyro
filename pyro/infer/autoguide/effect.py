# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Union

import torch
from torch.distributions import biject_to, constraints

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.effect_elbo import GuideMessenger
from pyro.nn.module import PyroModule, PyroParam, pyro_method
from pyro.poutine.runtime import get_plates

from .utils import deep_getattr, deep_setattr, helpful_support_errors


class AutoMessengerMeta(type(GuideMessenger), type(PyroModule)):
    pass


class AutoMessenger(GuideMessenger, PyroModule, metaclass=AutoMessengerMeta):
    """
    Base class for :class:`pyro.infer.effect_elbo.GuideMessenger` autoguides.
    """

    # Drop args for backwards compatibility with AutoGuide.
    def __init__(self, model, *, init_loc_fn=None):
        super().__init__()

    def __call__(self, *args, **kwargs):
        self._outer_plates = get_plates()
        return super().__call__(*args, **kwargs)

    def _remove_outer_plates(self, value, event_dim):
        """
        Removes particle plates from initial values of parameters.
        """
        for f in self._outer_plates:
            dim = -f.dim - event_dim
            if -value.dim() <= dim:
                dim = dim + value.dim()
                value = value[(slice(None),) * dim + slice(1)]
        for dim in range(value.dim() - event_dim):
            value = value.squeeze(0)
        return value


class AutoRegressiveMessenger(AutoMessenger):
    """
    Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` , intended for
    use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or similar.

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
                    scale = pyro.param("x_scale", lambda: torch.ones(prior.shape()))
                    return dist.Normal(loc, scale).to_event(prior.event_dim())
                # Fall back to autoregressive.
                return super().get_posterior(name, prior, upstream_values)
    """

    @pyro_method
    def get_posterior(
        self,
        name: str,
        prior: TorchDistribution,
        upstream_values: Dict[str, torch.Tensor],
    ) -> Union[TorchDistribution, torch.Tensor]:
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

    def _get_params(self, name, prior):
        try:
            loc = deep_getattr(self.locs, name)
            scale = deep_getattr(self.scales, name)
            return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with poutine.block(), torch.no_grad():
            constrained = prior.sample().detach()
            transform = biject_to(prior.support)
            unconstrained = transform.inv(constrained)
            event_dim = transform.domain.event_dim
            prototype = self._remove_outer_plates(unconstrained, event_dim)
        deep_setattr(
            self,
            "locs." + name,
            PyroParam(torch.zeros_like(prototype), event_dim=event_dim),
        )
        deep_setattr(
            self,
            "scales." + name,
            PyroParam(
                torch.ones_like(prototype),
                constraint=constraints.positive,
                event_dim=event_dim,
            ),
        )
        return self._get_params(name, prior)
