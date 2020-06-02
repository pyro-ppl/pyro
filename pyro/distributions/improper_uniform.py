# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from .torch_distribution import TorchDistribution
from .util import broadcast_shape


class ImproperUniform(TorchDistribution):
    """
    Improper distribution with zero :meth:`log_prob` and undefined
    :meth:`sample`.

    This is useful for transforming a model from generative dag form to factor
    graph form for use in HMC. For example the following are equal in
    distribution::

        # Version 1. a generative dag
        x = pyro.sample("x", Normal(0, 1))
        y = pyro.sample("y", Normal(x, 1))
        z = pyro.sample("z", Normal(y, 1))

        # Version 2. a factor graph
        xyz = pyro.sample("xyz", ImproperUniform(constraints.real, (), (3,)))
        x, y, z = xyz.unbind(-1)
        pyro.sample("x", Normal(0, 1), obs=x)
        pyro.sample("y", Normal(x, 1), obs=y)
        pyro.sample("z", Normal(y, 1), obs=z)

    Note this distribution errors when :meth:`sample` is called. To create a
    similar distribution that instead samples from a specified distribution
    consider using ``.mask(False)`` as in::

        xyz = dist.Normal(0, 1).expand([3]).to_event(1).mask(False)

    :param support: The support of the distribution.
    :type support: ~torch.distributions.constraints.Constraint
    :param torch.Size batch_shape: The batch shape.
    :param torch.Size event_shape: The event shape.
    """
    arg_constraints = {}

    def __init__(self, support, batch_shape, event_shape):
        assert isinstance(support, constraints.Constraint)
        self._support = support
        super().__init__(batch_shape, event_shape)

    @constraints.dependent_property
    def support(self):
        return self._support

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(ImproperUniform, _instance)
        new._support = self._support
        super(ImproperUniform, new).__init__(batch_shape, self.event_shape)
        return new

    def log_prob(self, value):
        batch_shape = value.shape[:value.dim() - self.event_dim]
        batch_shape = broadcast_shape(batch_shape, self.batch_shape)
        return torch.zeros(()).expand(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("ImproperUniform does not support sampling")
