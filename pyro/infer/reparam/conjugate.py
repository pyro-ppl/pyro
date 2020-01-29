# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .reparam import Reparam


class ConjugateReparam(Reparam):
    """
    Reparameterize to propose from a conjugate updated The likelihood function may be approximate or learned.

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

        # Draw a sample from the updated distribution.
        fn, log_normalizer = fn.conjugate_update(guide_dist)
        assert isinstance(guide_dist, dist.Distribution)
        if not fn.has_rsample:
            raise NotImplementedError("ConjugateReparam inference supports only reparameterized "
                                      "distributions, but got {}".format(type(fn)))
        value = pyro.sample("{}_posterior".format(name), fn,
                            infer={"_do_not_trace": True})

        # Return an importance-weighted point estimate.
        log_density = log_normalizer - guide_dist.log_prob(value)
        new_fn = dist.Delta(value, log_density=log_density, event_dim=fn.event_dim)
        return new_fn, value
