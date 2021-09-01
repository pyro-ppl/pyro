# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .reparam import Reparam


class StudentTReparam(Reparam):
    """
    Auxiliary variable reparameterizer for
    :class:`~pyro.distributions.StudentT` random variables.

    This is useful in combination with
    :class:`~pyro.infer.reparam.hmm.LinearHMMReparam` because it allows
    StudentT processes to be treated as conditionally Gaussian processes,
    permitting cheap inference via :class:`~pyro.distributions.GaussianHMM` .

    This reparameterizes a :class:`~pyro.distributions.StudentT` by introducing
    an auxiliary :class:`~pyro.distributions.Gamma` variable conditioned on
    which the result is :class:`~pyro.distributions.Normal` .
    """

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]

        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.StudentT)

        # Draw a sample that depends only on df.
        half_df = fn.df * 0.5
        gamma = pyro.sample(
            "{}_gamma".format(name), self._wrap(dist.Gamma(half_df, half_df), event_dim)
        )

        # Construct a scaled Normal.
        loc = fn.loc
        scale = fn.scale * gamma.rsqrt()
        new_fn = self._wrap(dist.Normal(loc, scale), event_dim)
        return {"fn": new_fn, "value": value, "is_observed": is_observed}
