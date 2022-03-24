# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

from torch.distributions import (
    Independent,
    MultivariateNormal,
    Normal,
    kl_divergence,
    register_kl,
)

from pyro.distributions.delta import Delta
from pyro.distributions.distribution import Distribution
from pyro.distributions.util import sum_rightmost


@register_kl(Delta, Distribution)
def _kl_delta(p, q):
    return -q.log_prob(p.v)


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    shared_ndims = min(p.reinterpreted_batch_ndims, q.reinterpreted_batch_ndims)
    p_ndims = p.reinterpreted_batch_ndims - shared_ndims
    q_ndims = q.reinterpreted_batch_ndims - shared_ndims
    p = Independent(p.base_dist, p_ndims) if p_ndims else p.base_dist
    q = Independent(q.base_dist, q_ndims) if q_ndims else q.base_dist
    kl = kl_divergence(p, q)
    if shared_ndims:
        kl = sum_rightmost(kl, shared_ndims)
    return kl


@register_kl(Independent, MultivariateNormal)
def _kl_independent_mvn(p, q):
    if isinstance(p.base_dist, Delta) and p.reinterpreted_batch_ndims == 1:
        return -q.log_prob(p.base_dist.v)

    if isinstance(p.base_dist, Normal) and p.reinterpreted_batch_ndims == 1:
        dim = q.event_shape[0]
        p_cov = p.base_dist.scale**2
        q_precision = q.precision_matrix.diagonal(dim1=-2, dim2=-1)
        return (
            0.5 * (p_cov * q_precision).sum(-1)
            - 0.5 * dim * (1 + math.log(2 * math.pi))
            - q.log_prob(p.base_dist.loc)
            - p.base_dist.scale.log().sum(-1)
        )

    raise NotImplementedError


__all__ = []
