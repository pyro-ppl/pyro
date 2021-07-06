# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist

from .reparam import Reparam


class GumbelSoftmaxReparam(Reparam):
    """
    Reparametrizer for :class:`~pyro.distributions.RelaxedOneHotCategorical`
    latent variables.

    This is useful for transforming multimodal posteriors to unimodal
    posteriors. Note this increases the latent dimension by 1 per event.

    This reparameterization works only for latent variables, not likelihoods.
    """

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]

        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.RelaxedOneHotCategorical)
        if is_observed:
            raise NotImplementedError(
                "SoftmaxReparam does not support observe statements"
                f" (at sample site {repr(name)})"
            )

        # Differentiably invert transform.
        u = None
        if value is not None:
            logits = value * fn.temperature
            u = (fn.logits - logits).exp().neg().exp()

        # Draw parameter-free noise.
        proto = fn.logits
        new_fn = dist.Uniform(torch.zeros_like(proto), torch.ones_like(proto))
        u = pyro.sample(
            f"{name}_uniform",
            self._wrap(new_fn, event_dim),
            obs=u,
            infer={"is_observed": is_observed},
        )

        # Differentiably transform.
        if value is None:
            logits = fn.logits - u.log().neg().log()
            value = (logits / fn.temperature).softmax(dim=-1)

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": value, "is_observed": True}
