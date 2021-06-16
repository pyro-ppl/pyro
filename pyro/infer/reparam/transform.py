# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .reparam import Reparam


class TransformReparam(Reparam):
    """
    Reparameterizer for
    :class:`pyro.distributions.torch.TransformedDistribution` latent variables.

    This is useful for transformed distributions with complex,
    geometry-changing transforms, where the posterior has simple shape in
    the space of ``base_dist``.

    This reparameterization works only for latent variables, not likelihoods.
    """
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.TransformedDistribution)

        # Differentiably invert transform.
        obs_base = obs
        if obs is not None:
            for t in reversed(fn.transforms):
                obs_base = t.inv(obs_base)

        # Draw noise from the base distribution.
        base_event_dim = event_dim
        for t in reversed(fn.transforms):
            base_event_dim += t.domain.event_dim - t.codomain.event_dim
        x = pyro.sample("{}_base".format(name),
                        self._wrap(fn.base_dist, base_event_dim),
                        obs=obs_base)

        # Differentiably transform.
        if obs is None:
            for t in fn.transforms:
                x = t(x)
        else:
            x = obs

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(x, event_dim=event_dim).mask(False)
        return new_fn, x
