# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist

from .reparam import Reparam


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
    """
    def __init__(self, sections, dim):
        assert isinstance(dim, int) and dim < 0
        assert isinstance(sections, list)
        assert all(isinstance(size, int) for size in sections)
        self.event_dim = -dim
        self.sections = sections

    def __call__(self, name, fn, obs):
        assert fn.event_dim >= self.event_dim
        assert obs is None, "SplitReparam does not support observe statements"

        # Draw independent parts.
        dim = fn.event_dim - self.event_dim
        left_shape = fn.event_shape[:dim]
        right_shape = fn.event_shape[1 + dim:]
        parts = []
        for i, size in enumerate(self.sections):
            event_shape = left_shape + (size,) + right_shape
            parts.append(pyro.sample(
                "{}_split_{}".format(name, i),
                dist.ImproperUniform(fn.support, fn.batch_shape, event_shape)))
        value = torch.cat(parts, dim=-self.event_dim)

        # Combine parts.
        log_prob = fn.log_prob(value)
        new_fn = dist.Delta(value, event_dim=fn.event_dim, log_density=log_prob)
        return new_fn, value
