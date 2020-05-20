# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import biject_to
from torch.distributions.transforms import ComposeTransform

import pyro
import pyro.distributions as dist

from .reparam import Reparam


# TODO Replace with .with_cache() once the following is released:
# https://github.com/probtorch/pytorch/pull/153
def _with_cache(t):
    return t.with_cache() if hasattr(t, "with_cache") else t


class UnitJacobianReparam(Reparam):
    """
    Reparameterizer for :class:`~torch.distributions.transforms.Transform`
    objects whose Jacobian determinant is one.

    :param transform: A transform whose Jacobian has determinant 1.
    :type transform: ~torch.distributions.transforms.Transform
    :param str suffix: A suffix to append to the transformed site.
    """
    def __init__(self, transform, suffix="transformed"):
        self.transform = _with_cache(transform)
        self.suffix = suffix

    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        assert fn.event_dim >= self.transform.event_dim, (
            "Cannot transform along batch dimension; "
            "try converting a batch dimension to an event dimension")

        # Draw noise from the base distribution.
        transform = ComposeTransform([_with_cache(biject_to(fn.support).inv),
                                      self.transform])
        x_trans = pyro.sample("{}_{}".format(name, self.suffix),
                              dist.TransformedDistribution(fn, transform))

        # Differentiably transform.
        x = transform.inv(x_trans)  # should be free due to transform cache

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(x, event_dim=fn.event_dim)
        return new_fn, x
