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

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]

        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.TransformedDistribution)

        # Differentiably invert transform.
        value_base = value
        if value is not None:
            for t in reversed(fn.transforms):
                value_base = t.inv(value_base)

        # Draw noise from the base distribution.
        base_event_dim = event_dim
        for t in reversed(fn.transforms):
            base_event_dim += t.domain.event_dim - t.codomain.event_dim
        value_base = pyro.sample(
            f"{name}_base",
            self._wrap(fn.base_dist, base_event_dim),
            obs=value_base,
            infer={"is_observed": is_observed},
        )

        # Differentiably transform.
        if value is None:
            value = value_base
            for t in fn.transforms:
                value = t(value)

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return {"fn": new_fn, "value": value, "is_observed": True}
