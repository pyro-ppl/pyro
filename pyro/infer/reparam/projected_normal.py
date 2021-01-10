# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist
from pyro.distributions.projected_normal import safe_project

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
        assert obs is None, "ProjectedNormalReparam does not support observe statements"

        # Draw parameter-free noise.
        new_fn = dist.Normal(fn.concentration, 1).to_event(1)
        x = pyro.sample("{}_normal".format(name), self._wrap(new_fn, event_dim))

        # Differentiably transform.
        value = safe_project(x)

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return new_fn, value
