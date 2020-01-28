# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .reparam import Reparam


class ConjugateReparam(Reparam):
    """
    Reparameterize to propose from the conjugate posterior given a likelihood
    function.  The likelihood function may be approximate or learned.

    :param guide: A likelihood distribution or a callable returning a
        likelihood distribution.
    :type guide: ~pyro.distributions.Distribution or callable
    """
    def __init__(self, guide):
        self.guide = guide

    def __call__(self, name, fn, obs):
        assert obs is None, "PosteriorReparam does not support observe statements"

        # Create a likelihood guide.
        guide_dist = self.guide
        if not isinstance(guide_dist, dist.Distribution):
            args, kwargs = self.args_kwargs
            guide_dist = guide_dist(*args, **kwargs)
        assert isinstance(guide_dist, dist.Distribution)

        # Draw a sample from the approximate posterior.
        posterior, log_normalizer = fn.posterior(guide_dist)
        assert isinstance(guide_dist, dist.Distribution)
        value = pyro.sample("{}_posterior", posterior, infer={"require_guide": False})

        # Return an importance-weighted point estimate.
        # This is equal to fn.log_prob(value) - posterior.log_prob(value).
        log_density = log_normalizer - guide_dist.log_prob(value)
        new_fn = dist.Delta(value, log_density=log_density, event_dim=fn.event_dim)
        return new_fn, value
