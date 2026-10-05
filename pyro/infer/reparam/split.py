# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.ops.tensor_utils import broadcast_tensors_without_dim

from .reparam import Reparam


def same_support(fn: TorchDistributionMixin, *args):
    """
    Returns support of the `fn` distribution. Used in :class:`SplitReparam` in
    order to determine the support of the split value.

    :param fn: distribution class
    :returns: distribution support
    """
    return fn.support


def real_support(fn: TorchDistributionMixin, *args):
    """
    Returns real support with same event dimension as that of the `fn` distribution.
    Used in :class:`SplitReparam` in order to determine the support of the split value.

    :param fn: distribution class
    :returns: distribution support
    """
    return dist.constraints.independent(dist.constraints.real, fn.event_dim)


def default_support(fn: TorchDistributionMixin, slice, dim):
    """
    Returns support of the `fn` distribution, corrected for split stacking and
    concatenation transforms. Used in :class:`SplitReparam` in
    order to determine the support of the split value.

    :param fn: distribution class
    :param slice: slice for which to return support
    :param dim: dimension for which to return support
    :returns: distribution support
    """
    support = fn.support
    # Unwrap support
    reinterpreted_batch_ndims_vec = []
    while isinstance(support, dist.constraints.independent):
        reinterpreted_batch_ndims_vec.append(support.reinterpreted_batch_ndims)
        support = support.base_constraint
    # Slice concatenation and stacking transforms
    if isinstance(support, dist.constraints.stack) and support.dim == dim:
        support = dist.constraints.stack(support.cseq[slice], dim)
    elif isinstance(support, dist.constraints.cat) and support.dim == dim:
        support = dist.constraints.cat(support.cseq[slice], dim, support.lengths[slice])
    # Wrap support
    for reinterpreted_batch_ndims in reinterpreted_batch_ndims_vec[::-1]:
        support = dist.constraints.independent(support, reinterpreted_batch_ndims)
    return support


class SplitReparam(Reparam):
    """
    Reparameterizer to split a random variable along a dimension, similar to
    :func:`torch.split`.

    This is useful for treating different parts of a tensor with different
    reparameterizers or inference methods. For example when performing HMC
    inference on a time series, you can first apply
    :class:`~pyro.infer.reparam.discrete_cosine.DiscreteCosineReparam` or
    :class:`~pyro.infer.reparam.haar.HaarReparam`, then apply
    :class:`SplitReparam` to split into low-frequency and high-frequency
    components, and finally add the low-frequency components to the
    ``full_mass`` matrix together with globals.

    :param sections: Size of a single chunk or list of sizes for
        each chunk.
    :type: list(int)
    :param int dim: Dimension along which to split. Defaults to -1.
    :param callable support_fn: Function which derives the split support
        from the site's sampling function, split size, and split dimension.
        Default is :func:`default_support` which correctly handles stacking
        and concatenation transforms. Other options are :func:`same_support`
        which returns the same support as that of the sampling function, and
        :func:`real_support` which returns a real support.
    """

    def __init__(self, sections, dim, support_fn=default_support):
        assert isinstance(dim, int) and dim < 0
        assert isinstance(sections, list)
        assert all(isinstance(size, int) for size in sections)
        self.event_dim = -dim
        self.sections = sections
        self.support_fn = support_fn

    def apply(self, msg):
        name = msg["name"]
        fn = msg["fn"]
        value = msg["value"]
        is_observed = msg["is_observed"]
        assert fn.event_dim >= self.event_dim

        # Split value into parts.
        value_split = [None] * len(self.sections)
        if value is not None:
            value_split[:] = value.split(self.sections, -self.event_dim)

        # Draw independent parts.
        dim = fn.event_dim - self.event_dim
        left_shape = fn.event_shape[:dim]
        right_shape = fn.event_shape[1 + dim :]
        start = 0
        for i, size in enumerate(self.sections):
            event_shape = left_shape + (size,) + right_shape
            value_split[i] = pyro.sample(
                f"{name}_split_{i}",
                dist.ImproperUniform(
                    self.support_fn(fn, slice(start, start + size), -self.event_dim),
                    fn.batch_shape,
                    event_shape,
                ),
                obs=value_split[i],
                infer={"is_observed": is_observed},
            )
            start += size

        # Combine parts into value.
        if value is None:
            value_split = broadcast_tensors_without_dim(value_split, -self.event_dim)
            value = torch.cat(value_split, dim=-self.event_dim)

        if poutine.get_mask() is False:
            log_density = 0.0
        else:
            log_density = fn.log_prob(value)
        new_fn = dist.Delta(value, event_dim=fn.event_dim, log_density=log_density)
        return {"fn": new_fn, "value": value, "is_observed": True}
