# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .reparam import Reparam


class ConjugateReparam(Reparam):
    """
    EXPERIMENTAL Reparameterize to a conjugate updated distribution.

    This updates a prior distribution ``fn`` using the
    :meth:`~pyro.distributions.Distribution.conjugate_update`
    method.  The guide may be either a distribution object or a callable
    inputting model ``*args,**kwargs`` and returning a distribution object. The
    guide may be approximate or learned.

    For example consider the model and naive variational guide::

        total = torch.tensor(10.)
        count = torch.tensor(2.)

        def model():
            prob = pyro.sample("prob", dist.Beta(0.5, 1.5))
            pyro.sample("count", dist.Binomial(total, prob), obs=count)

        guide = AutoDiagonalNormal(model)  # learns the posterior over prob

    Instead of using this learned guide, we can hand-compute the conjugate
    posterior distribution over "prob", and then use a simpler guide during
    inference, in this case an empty guide::

        reparam_model = poutine.reparam(model, {
            "prob": ConjugateReparam(dist.Beta(1 + count, 1 + total - count))
        })

        def reparam_guide():
            pass  # nothing remains to be modeled!

    :param guide: A likelihood distribution or a callable returning a
        guide distribution. Only a few distributions are supported, depending
        on the prior distribution's
        :meth:`~pyro.distributions.Distribution.conjugate_update`
        implementation.
    :type guide: ~pyro.distributions.Distribution or callable
    """
    def __init__(self, guide):
        self.guide = guide

    def __call__(self, name, fn, obs):
        assert obs is None, "PosteriorReparam does not support observe statements"

        # Compute a guide distribution, either static or dependent.
        guide_dist = self.guide
        if not isinstance(guide_dist, dist.Distribution):
            args, kwargs = self.args_kwargs
            guide_dist = guide_dist(*args, **kwargs)
        assert isinstance(guide_dist, dist.Distribution)

        # Draw a sample from the updated distribution.
        fn, log_normalizer = fn.conjugate_update(guide_dist)
        assert isinstance(guide_dist, dist.Distribution)
        if not fn.has_rsample:
            # Note supporting non-reparameterized sites would require more delicate
            # handling of traced sites than the crude _do_not_trace flag below.
            raise NotImplementedError("ConjugateReparam inference supports only reparameterized "
                                      "distributions, but got {}".format(type(fn)))
        value = pyro.sample("{}_updated".format(name), fn,
                            infer={"is_auxiliary": True, "_do_not_trace": True})

        # Compute importance weight. Let p(z) be the original fn, q(z|x) be
        # the guide, and u(z) be the conjugate_updated distribution. Then
        #   normalizer = p(z) q(z|x) / u(z).
        # Since we've sampled from u(z) instead of p(z), we
        # need an importance weight
        #   p(z) / u(z) = normalizer / q(z|x)                          (Eqn 1)
        # Note that q(z|x) is often approximate; in the exact case
        #   q(z|x) = p(x|z) / integral p(x|z) dz
        # so this site and the downstream likelihood site will have combined density
        #   (p(z) / u(z)) p(x|z) = (normalizer / q(z|x)) p(x|z)
        #                        = normalizer integral p(x|z) dz
        # Hence in the exact case, downstream probability does not depend on the sampled z,
        # permitting this reparameterizer to be used in HMC.
        log_density = log_normalizer - guide_dist.log_prob(value)  # By Eqn 1.

        # Return an importance-weighted point estimate.
        new_fn = dist.Delta(value, log_density=log_density, event_dim=fn.event_dim)
        return new_fn, value
