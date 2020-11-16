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
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.RelaxedOneHotCategorical)
        assert obs is None, "SoftmaxReparam does not support observe statements"

        # Draw parameter-free noise.
        proto = fn.logits
        new_fn = dist.Uniform(torch.zeros_like(proto), torch.ones_like(proto))
        u = pyro.sample("{}_uniform".format(name), self._wrap(new_fn, event_dim))

        # Differentiably transform.
        logits = fn.logits - u.log().neg().log()
        value = (logits / fn.temperature).softmax(dim=-1)

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return new_fn, value
