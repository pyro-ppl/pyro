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
        assert obs is None, "TransformReparam does not support observe statements"
        assert isinstance(fn, dist.TransformedDistribution)

        # Draw noise from the base distribution.
        x = pyro.sample("{}_base".format(name), fn.base_dist)

        # Differentiably transform.
        for t in fn.transforms:
            x = t(x)

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(x, event_dim=fn.event_dim)
        return new_fn, x
