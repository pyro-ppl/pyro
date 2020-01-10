# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform

from pyro.distributions.torch import TransformedDistribution


class FoldedDistribution(TransformedDistribution):
    """
    Equivalent to ``TransformedDistribution(base_dist, AbsTransform())``,
    but additionally supports :meth:`log_prob` .

    :param ~torch.distributions.Distribution base_dist: The distribution to
        reflect.
    """
    support = constraints.positive

    def __init__(self, base_dist, validate_args=None):
        if base_dist.event_shape:
            raise ValueError("Only univariate distributions can be folded.")
        super().__init__(base_dist, AbsTransform(), validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        return super().expand(batch_shape, _instance=new)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dim = max(len(self.batch_shape), value.dim())
        plus_minus = value.new_tensor([1., -1.]).reshape((2,) + (1,) * dim)
        return self.base_dist.log_prob(plus_minus * value).logsumexp(0)
