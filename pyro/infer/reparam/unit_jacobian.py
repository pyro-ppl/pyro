# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

from torch.distributions import biject_to
from torch.distributions.transforms import ComposeTransform

import pyro
import pyro.distributions as dist
from pyro.poutine.plate_messenger import block_plate

from .reparam import Reparam


class UnitJacobianReparam(Reparam):
    """
    Reparameterizer for :class:`~torch.distributions.transforms.Transform`
    objects whose Jacobian determinant is one.

    :param transform: A transform whose Jacobian has determinant 1.
    :type transform: ~torch.distributions.transforms.Transform
    :param str suffix: A suffix to append to the transformed site.
    :param bool experimental_allow_batch: EXPERIMENTAL allow coupling across a
        batch dimension. The targeted batch dimension and all batch dimensions
        to the right will be converted to event dimensions. Defaults to False.
    """

    def __init__(
        self, transform, suffix="transformed", *, experimental_allow_batch=False
    ):
        self.transform = transform.with_cache()
        self.suffix = suffix
        self.experimental_allow_batch = experimental_allow_batch

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]

        event_dim = fn.event_dim
        transform = self.transform
        with ExitStack() as stack:
            shift = max(0, transform.event_dim - event_dim)
            if shift:
                if not self.experimental_allow_batch:
                    raise ValueError(
                        "Cannot transform along batch dimension; try either"
                        "converting a batch dimension to an event dimension, or "
                        "setting experimental_allow_batch=True."
                    )

                # Reshape and mute plates using block_plate.
                from pyro.contrib.forecast.util import (
                    reshape_batch,
                    reshape_transform_batch,
                )

                old_shape = fn.batch_shape
                new_shape = old_shape[:-shift] + (1,) * shift + old_shape[-shift:]
                fn = reshape_batch(fn, new_shape).to_event(shift)
                transform = reshape_transform_batch(
                    transform, old_shape + fn.event_shape, new_shape + fn.event_shape
                )
                if value is not None:
                    value = value.reshape(
                        value.shape[: -shift - event_dim]
                        + (1,) * shift
                        + value.shape[-shift - event_dim :]
                    )
                for dim in range(-shift, 0):
                    stack.enter_context(block_plate(dim=dim, strict=False))

            # Differentiably invert transform.
            transform = ComposeTransform(
                [biject_to(fn.support).inv.with_cache(), self.transform]
            )
            value_trans = None
            if value is not None:
                value_trans = transform(value)

            # Draw noise from the base distribution.
            value_trans = pyro.sample(
                f"{name}_{self.suffix}",
                dist.TransformedDistribution(fn, transform),
                obs=value_trans,
                infer={"is_observed": is_observed},
            )

        # Differentiably transform. This should be free due to transform cache.
        if value is None:
            value = transform.inv(value_trans)
        if shift:
            value = value.reshape(
                value.shape[: -2 * shift - event_dim]
                + value.shape[-shift - event_dim :]
            )

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim)
        return {"fn": new_fn, "value": value, "is_observed": True}
