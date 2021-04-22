# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro import poutine
from pyro.infer.autoguide.guides import AutoStructured

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
        for p in guide.parameters():
            p.detach_()
        self.guide = guide
        self.deltas = {}

    def _reparam_config(self, site):
        if site["name"] in self.guide.prototype_trace:
            return self

    def reparam(self, fn=None):
        return poutine.reparam(fn, config=self._reparam_config)

    def __call__(self, name, fn, obs):
        assert obs is None, "StructuredReparam does not support observe statements"
        if not self.deltas:  # On first sample site.
            self.deltas = self.guide.get_deltas()
        new_fn = self.deltas.pop(name)
        value = new_fn.v
        return new_fn, value
