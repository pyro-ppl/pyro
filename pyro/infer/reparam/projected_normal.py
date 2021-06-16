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
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.ProjectedNormal)

        # Differentiably invert transform.
        obs_normal = None
        if obs is not None:
            obs_normal = obs - fn.concentration

        # Draw parameter-free noise.
        new_fn = dist.Normal(torch.zeros_like(fn.concentration), 1).to_event(1)
        x = pyro.sample("{}_normal".format(name), self._wrap(new_fn, event_dim),
                        obs=obs_normal)

        # Differentiably transform.
        value = safe_normalize(x + fn.concentration) if obs is None else obs

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return new_fn, value
