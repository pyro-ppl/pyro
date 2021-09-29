# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide.structured import AutoStructured
from pyro.poutine.plate_messenger import block_plate

from .reparam import Reparam


class StructuredReparam(Reparam):
    """
    Preconditioning reparameterizer of multiple latent variables.

    This uses a trained :class:`~pyro.infer.autoguide.AutoStructured`
    guide to alter the geometry of a model, typically for use e.g. in MCMC.
    Example usage::

        # Step 1. Train a guide
        guide = AutoStructured(model, ...)
        svi = SVI(model, guide, ...)
        # ...train the guide...

        # Step 2. Use trained guide in preconditioned MCMC
        model = StructuredReparam(guide).reparam(model)
        nuts = NUTS(model)
        # ...now use the model in HMC or NUTS...

    This reparameterization works only for latent variables, not likelihoods.
    Note that all sites must share a single common :class:`StructuredReparam`
    instance, and that the model must have static structure.

    .. note:: This can be seen as a restricted structured version of
        :class:`~pyro.infer.reparam.neutra.NeuTraReparam` [1] combined with
        ``poutine.condition`` on MAP-estimated sites (the NeuTra transform is
        an exact reparameterizer, but the conditioning to point estimates
        introduces model approximation).

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704

    :param ~pyro.infer.autoguide.AutoStructured guide: A trained guide.
    """

    def __init__(self, guide: AutoStructured):
        if not isinstance(guide, AutoStructured):
            raise TypeError(
                f"StructuredReparam expected an AutoStructured guide, but got {type(guide)}"
            )
        self.guide = guide.requires_grad_(False)
        self.deltas = {}

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
                "StructuredReparam does not support observe statements"
            )

        if name not in self.deltas:  # On first sample site.
            with ExitStack() as stack:
                for plate in self.guide.plates.values():
                    stack.enter_context(block_plate(dim=plate.dim, strict=False))
                self.deltas = self.guide.get_deltas()
        new_fn = self.deltas.pop(name)
        value = new_fn.v

        if poutine.get_mask() is not False:
            log_density = new_fn.log_density + fn.log_prob(value)
            new_fn = dist.Delta(value, log_density, new_fn.event_dim)
        return {"fn": new_fn, "value": value, "is_observed": True}

    def transform_samples(self, aux_samples, save_params=None):
        """
        Given latent samples from the warped posterior (with a possible batch dimension),
        return a `dict` of samples from the latent sites in the model.

        :param dict aux_samples: Dict site name to tensor value for each latent
            auxiliary site (or if ``save_params`` is specifiec, then for only
            those latent auxiliary sites needed to compute requested params).
        :param list save_params: An optional list of site names to save. This
            is useful in models with large nuisance variables. Defaults to
            None, saving all params.
        :return: a `dict` of samples keyed by latent sites in the model.
        :rtype: dict
        """
        with poutine.condition(data=aux_samples), poutine.mask(mask=False):
            deltas = self.guide.get_deltas(save_params)
        return {name: delta.v for name, delta in deltas.items()}
