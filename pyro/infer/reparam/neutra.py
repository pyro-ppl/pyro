# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.util import sum_rightmost
from pyro.infer.autoguide.guides import AutoContinuous
from pyro.poutine.plate_messenger import block_plate

from .reparam import Reparam


class NeuTraReparam(Reparam):
    """
    Neural Transport reparameterizer [1] of multiple latent variables.

    This uses a trained :class:`~pyro.infer.autoguide.AutoContinuous`
    guide to alter the geometry of a model, typically for use e.g. in MCMC.
    Example usage::

        # Step 1. Train a guide
        guide = AutoIAFNormal(model)
        svi = SVI(model, guide, ...)
        # ...train the guide...

        # Step 2. Use trained guide in NeuTra MCMC
        neutra = NeuTraReparam(guide)
        model = poutine.reparam(model, config=lambda _: neutra)
        nuts = NUTS(model)
        # ...now use the model in HMC or NUTS...

    This reparameterization works only for latent variables, not likelihoods.
    Note that all sites must share a single common :class:`NeuTraReparam`
    instance, and that the model must have static structure.

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704

    :param ~pyro.infer.autoguide.AutoContinuous guide: A trained guide.
    """

    def __init__(self, guide):
        if not isinstance(guide, AutoContinuous):
            raise TypeError(
                "NeuTraReparam expected an AutoContinuous guide, but got {}".format(
                    type(guide)
                )
            )
        self.guide = guide
        self.transform = None
        self.x_unconstrained = {}
        self.is_observed = None

    def _reparam_config(self, site):
        if site["name"] in self.guide.prototype_trace:
            return self

    def reparam(self, fn=None):
        return poutine.reparam(fn, config=self._reparam_config)

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]
        if name not in self.guide.prototype_trace.nodes:
            return {"fn": fn, "value": value, "is_observed": is_observed}
        if is_observed:
            raise NotImplementedError(
                f"At pyro.sample({repr(name)},...), "
                "NeuTraReparam does not support observe statements."
            )

        log_density = 0.0
        compute_density = poutine.get_mask() is not False
        if name not in self.x_unconstrained:  # On first sample site.
            # Sample a shared latent.
            try:
                self.transform = self.guide.get_transform()
            except (NotImplementedError, TypeError) as e:
                raise ValueError(
                    "NeuTraReparam only supports guides that implement "
                    "`get_transform` method that does not depend on the "
                    "model's `*args, **kwargs`"
                ) from e

            with ExitStack() as stack:
                for plate in self.guide.plates.values():
                    stack.enter_context(block_plate(dim=plate.dim, strict=False))
                z_unconstrained = pyro.sample(
                    f"{name}_shared_latent", self.guide.get_base_dist().mask(False)
                )

            # Differentiably transform.
            x_unconstrained = self.transform(z_unconstrained)
            if compute_density:
                log_density = self.transform.log_abs_det_jacobian(
                    z_unconstrained, x_unconstrained
                )
            self.x_unconstrained = {
                site["name"]: (site, unconstrained_value)
                for site, unconstrained_value in self.guide._unpack_latent(
                    x_unconstrained
                )
            }

        # Extract a single site's value from the shared latent.
        site, unconstrained_value = self.x_unconstrained.pop(name)
        transform = biject_to(fn.support)
        value = transform(unconstrained_value)
        if compute_density:
            logdet = transform.log_abs_det_jacobian(unconstrained_value, value)
            logdet = sum_rightmost(logdet, logdet.dim() - value.dim() + fn.event_dim)
            log_density = log_density + fn.log_prob(value) + logdet
        new_fn = dist.Delta(value, log_density, event_dim=fn.event_dim)
        return {"fn": new_fn, "value": value, "is_observed": True}

    def transform_sample(self, latent):
        """
        Given latent samples from the warped posterior (with a possible batch dimension),
        return a `dict` of samples from the latent sites in the model.

        :param latent: sample from the warped posterior (possibly batched). Note that the
            batch dimension must not collide with plate dimensions in the model, i.e.
            any batch dims `d < - max_plate_nesting`.
        :return: a `dict` of samples keyed by latent sites in the model.
        :rtype: dict
        """
        x_unconstrained = self.transform(latent)
        transformed_samples = {}
        for site, value in self.guide._unpack_latent(x_unconstrained):
            transform = biject_to(site["fn"].support)
            x_constrained = transform(value)
            transformed_samples[site["name"]] = x_constrained
        return transformed_samples
