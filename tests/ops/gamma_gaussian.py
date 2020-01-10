# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro.distributions as dist
from pyro.ops.gamma_gaussian import GammaGaussian
from tests.common import assert_close


def random_gamma_gaussian(batch_shape, dim, rank=None):
    """
    Generate a random Gaussian for testing.
    """
    if rank is None:
        rank = dim + dim
    log_normalizer = torch.randn(batch_shape)
    loc = torch.randn(batch_shape + (dim,))
    samples = torch.randn(batch_shape + (dim, rank))
    precision = torch.matmul(samples, samples.transpose(-2, -1))
    if dim > 0:
        info_vec = precision.matmul(loc.unsqueeze(-1)).squeeze(-1)
    else:
        info_vec = loc
    alpha = torch.randn(batch_shape).exp() + 0.5 * dim - 1
    beta = torch.randn(batch_shape).exp() + 0.5 * (info_vec * loc).sum(-1)
    result = GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)
    assert result.dim() == dim
    assert result.batch_shape == batch_shape
    return result


def random_gamma(batch_shape):
    """
    Generate a random Gamma distribution for testing.
    """
    concentration = torch.randn(batch_shape).exp()
    rate = torch.randn(batch_shape).exp()
    return dist.Gamma(concentration, rate)


def assert_close_gamma_gaussian(actual, expected):
    assert isinstance(actual, GammaGaussian)
    assert isinstance(expected, GammaGaussian)
    assert actual.dim() == expected.dim()
    assert actual.batch_shape == expected.batch_shape
    assert_close(actual.log_normalizer, expected.log_normalizer)
    assert_close(actual.info_vec, expected.info_vec)
    assert_close(actual.precision, expected.precision)
    assert_close(actual.alpha, expected.alpha)
    assert_close(actual.beta, expected.beta)
