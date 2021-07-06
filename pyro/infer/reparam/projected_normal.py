# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.ops.tensor_utils import safe_normalize

from .reparam import Reparam


class ProjectedNormalReparam(Reparam):
    """
    Reparametrizer for :class:`~pyro.distributions.ProjectedNormal` latent
    variables.

    This reparameterization works only for latent variables, not likelihoods.
    """

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]
        if is_observed:
            raise NotImplementedError(
                "ProjectedNormalReparam does not support observe statements"
            )

        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.ProjectedNormal)

        # Differentiably invert transform.
        value_normal = None
        if value is not None:
            # We use an arbitrary injection, which works only for initialization.
            value_normal = value - fn.concentration

        # Draw parameter-free noise.
        new_fn = dist.Normal(torch.zeros_like(fn.concentration), 1).to_event(1)
        x = pyro.sample(
            "{}_normal".format(name),
            self._wrap(new_fn, event_dim),
            obs=value_normal,
            infer={"is_observed": is_observed},
        )

        # Differentiably transform.
        if value is None:
            value = safe_normalize(x + fn.concentration)

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": value, "is_observed": True}
