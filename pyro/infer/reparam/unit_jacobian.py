# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

from torch.distributions import biject_to
from torch.distributions.transforms import ComposeTransform

import pyro
import pyro.distributions as dist
from pyro.poutine.plate_messenger import block_plate

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
    def __init__(self, transform, suffix="transformed", *,
                 experimental_allow_batch=False):
        self.transform = _with_cache(transform)
        self.suffix = suffix
        self.experimental_allow_batch = experimental_allow_batch

    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        event_dim = fn.event_dim
        transform = self.transform
        with ExitStack() as stack:
            shift = max(0, transform.event_dim - event_dim)
            if shift:
                assert self.experimental_allow_batch, (
                    "Cannot transform along batch dimension; "
                    "try converting a batch dimension to an event dimension")

                # Reshape and mute plates using block_plate.
                from pyro.contrib.forecast.util import reshape_batch, reshape_transform_batch
                old_shape = fn.batch_shape
                new_shape = old_shape[:-shift] + (1,) * shift + old_shape[-shift:]
                fn = reshape_batch(fn, new_shape).to_event(shift)
                transform = reshape_transform_batch(transform,
                                                    old_shape + fn.event_shape,
                                                    new_shape + fn.event_shape)
                for dim in range(-shift, 0):
                    stack.enter_context(block_plate(dim=dim))

            # Draw noise from the base distribution.
            transform = ComposeTransform([_with_cache(biject_to(fn.support).inv),
                                          self.transform])
            x_trans = pyro.sample("{}_{}".format(name, self.suffix),
                                  dist.TransformedDistribution(fn, transform))

        # Differentiably transform.
        x = transform.inv(x_trans)  # should be free due to transform cache
        if shift:
            x = x.reshape(x.shape[:-2 * shift - event_dim] + x.shape[-shift - event_dim:])

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(x, event_dim=event_dim)
        return new_fn, x
