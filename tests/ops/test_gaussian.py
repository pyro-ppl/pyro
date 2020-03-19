# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.nn.functional import pad

import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.ops.gaussian import AffineNormal, Gaussian, gaussian_tensordot, matrix_and_mvn_to_gaussian, mvn_to_gaussian
from tests.common import assert_close
from tests.ops.gaussian import assert_close_gaussian, random_gaussian, random_mvn


@pytest.mark.parametrize("extra_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("log_normalizer_shape,info_vec_shape,precision_shape", [
    ((), (), ()),
    ((5,), (), ()),
    ((), (5,), ()),
    ((), (), (5,)),
    ((3, 1, 1), (1, 4, 1), (1, 1, 5)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_expand(extra_shape, log_normalizer_shape, info_vec_shape, precision_shape, dim):
    rank = dim + dim
    log_normalizer = torch.randn(log_normalizer_shape)
    info_vec = torch.randn(info_vec_shape + (dim,))
    precision = torch.randn(precision_shape + (dim, rank))
    precision = precision.matmul(precision.transpose(-1, -2))
    gaussian = Gaussian(log_normalizer, info_vec, precision)

    expected_shape = extra_shape + broadcast_shape(
        log_normalizer_shape, info_vec_shape, precision_shape)
    actual = gaussian.expand(expected_shape)
    assert actual.batch_shape == expected_shape


@pytest.mark.parametrize("old_shape,new_shape", [
    ((6,), (3, 2)),
    ((5, 6), (5, 3, 2)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_reshape(old_shape, new_shape, dim):
    gaussian = random_gaussian(old_shape, dim)

    # reshape to new
    new = gaussian.reshape(new_shape)
    assert new.batch_shape == new_shape

    # reshape back to old
    g = new.reshape(old_shape)
    assert_close_gaussian(g, gaussian)


@pytest.mark.parametrize("shape,cat_dim,split", [
    ((4, 7, 6), -1, (2, 1, 3)),
    ((4, 7, 6), -2, (1, 1, 2, 3)),
    ((4, 7, 6), 1, (1, 1, 2, 3)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_cat(shape, cat_dim, split, dim):
    assert sum(split) == shape[cat_dim]
    gaussian = random_gaussian(shape, dim)
    parts = []
    end = 0
    for size in split:
        beg, end = end, end + size
        if cat_dim == -1:
            part = gaussian[..., beg: end]
        elif cat_dim == -2:
            part = gaussian[..., beg: end, :]
        elif cat_dim == 1:
            part = gaussian[:, beg: end]
        else:
            raise ValueError
        parts.append(part)

    actual = Gaussian.cat(parts, cat_dim)
    assert_close_gaussian(actual, gaussian)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("left", [0, 1, 2])
@pytest.mark.parametrize("right", [0, 1, 2])
def test_pad(shape, left, right, dim):
    expected = random_gaussian(shape, dim)
    padded = expected.event_pad(left=left, right=right)
    assert padded.batch_shape == expected.batch_shape
    assert padded.dim() == left + expected.dim() + right
    mid = slice(left, padded.dim() - right)
    assert_close(padded.info_vec[..., mid], expected.info_vec)
    assert_close(padded.precision[..., mid, mid], expected.precision)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_add(shape, dim):
    x = random_gaussian(shape, dim)
    y = random_gaussian(shape, dim)
    value = torch.randn(dim)
    assert_close((x + y).log_density(value), x.log_density(value) + y.log_density(value))


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_rsample_shape(sample_shape, batch_shape, dim):
    mvn = random_mvn(batch_shape, dim)
    g = mvn_to_gaussian(mvn)
    expected = mvn.rsample(sample_shape)
    actual = g.rsample(sample_shape)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_rsample_distribution(batch_shape, dim):
    num_samples = 20000
    mvn = random_mvn(batch_shape, dim)
    g = mvn_to_gaussian(mvn)
    expected = mvn.rsample((num_samples,))
    actual = g.rsample((num_samples,))

    def get_moments(x):
        mean = x.mean(0)
        x = x - mean
        cov = (x.unsqueeze(-1) * x.unsqueeze(-2)).mean(0)
        std = cov.diagonal(dim1=-1, dim2=-2).sqrt()
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2))
        return mean, std, corr

    expected_mean, expected_std, expected_corr = get_moments(expected)
    actual_mean, actual_std, actual_corr = get_moments(actual)
    assert_close(actual_mean, expected_mean, atol=0.1, rtol=0.02)
    assert_close(actual_std, expected_std, atol=0.1, rtol=0.02)
    assert_close(actual_corr, expected_corr, atol=0.05)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize_shape(batch_shape, left, right):
    dim = left + right
    g = random_gaussian(batch_shape, dim)
    assert g.marginalize(left=left).dim() == right
    assert g.marginalize(right=right).dim() == left


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize(batch_shape, left, right):
    dim = left + right
    g = random_gaussian(batch_shape, dim)
    assert_close(g.marginalize(left=left).event_logsumexp(),
                 g.event_logsumexp())
    assert_close(g.marginalize(right=right).event_logsumexp(),
                 g.event_logsumexp())


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize_condition(sample_shape, batch_shape, left, right):
    dim = left + right
    g = random_gaussian(batch_shape, dim)
    x = torch.randn(sample_shape + (1,) * len(batch_shape) + (right,))
    assert_close(g.marginalize(left=left).log_density(x),
                 g.condition(x).event_logsumexp())


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_condition(sample_shape, batch_shape, left, right):
    dim = left + right
    gaussian = random_gaussian(batch_shape, dim)
    gaussian.precision += torch.eye(dim) * 0.1
    value = torch.randn(sample_shape + (1,) * len(batch_shape) + (dim,))
    left_value, right_value = value[..., :left], value[..., left:]

    conditioned = gaussian.condition(right_value)
    assert conditioned.batch_shape == sample_shape + gaussian.batch_shape
    assert conditioned.dim() == left

    actual = conditioned.log_density(left_value)
    expected = gaussian.log_density(value)
    assert_close(actual, expected)

    # test left_condition
    permute_conditioned = gaussian.left_condition(left_value)
    assert permute_conditioned.batch_shape == sample_shape + gaussian.batch_shape
    assert permute_conditioned.dim() == right

    permute_actual = permute_conditioned.log_density(right_value)
    assert_close(permute_actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_logsumexp(batch_shape, dim):
    gaussian = random_gaussian(batch_shape, dim)
    gaussian.info_vec *= 0.1  # approximately centered
    gaussian.precision += torch.eye(dim) * 0.1

    num_samples = 200000
    scale = 10
    samples = torch.rand((num_samples,) + (1,) * len(batch_shape) + (dim,)) * scale - scale / 2
    expected = gaussian.log_density(samples).logsumexp(0) + math.log(scale ** dim / num_samples)
    actual = gaussian.event_logsumexp()
    assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("x_dim", [1, 2, 3])
@pytest.mark.parametrize("y_dim", [1, 2, 3])
def test_affine_normal(batch_shape, x_dim, y_dim):
    matrix = torch.randn(batch_shape + (x_dim, y_dim))
    loc = torch.randn(batch_shape + (y_dim,))
    scale = torch.randn(batch_shape + (y_dim,)).exp()
    y = torch.randn(batch_shape + (y_dim,))

    normal = dist.Normal(loc, scale).to_event(1)
    actual = matrix_and_mvn_to_gaussian(matrix, normal)
    assert isinstance(actual, AffineNormal)
    actual_like = actual.condition(y)
    assert isinstance(actual_like, Gaussian)

    mvn = dist.MultivariateNormal(loc, scale_tril=scale.diag_embed())
    expected = matrix_and_mvn_to_gaussian(matrix, mvn)
    assert isinstance(expected, Gaussian)
    expected_like = expected.condition(y)
    assert isinstance(expected_like, Gaussian)

    assert_close(actual_like.log_normalizer, expected_like.log_normalizer)
    assert_close(actual_like.info_vec, expected_like.info_vec)
    assert_close(actual_like.precision, expected_like.precision)

    x = torch.randn(batch_shape + (x_dim,))
    permute_actual = actual.left_condition(x)
    assert isinstance(permute_actual, AffineNormal)
    permute_actual = permute_actual.to_gaussian()

    permute_expected = expected.left_condition(y)
    assert isinstance(permute_expected, Gaussian)

    assert_close(permute_actual.log_normalizer, permute_actual.log_normalizer)
    assert_close(permute_actual.info_vec, permute_actual.info_vec)
    assert_close(permute_actual.precision, permute_actual.precision)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_mvn_to_gaussian(sample_shape, batch_shape, dim):
    mvn = random_mvn(batch_shape, dim)
    gaussian = mvn_to_gaussian(mvn)
    value = mvn.sample(sample_shape)
    actual_log_prob = gaussian.log_density(value)
    expected_log_prob = mvn.log_prob(value)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("x_dim", [1, 2, 3])
@pytest.mark.parametrize("y_dim", [1, 2, 3])
def test_matrix_and_mvn_to_gaussian(sample_shape, batch_shape, x_dim, y_dim):
    matrix = torch.randn(batch_shape + (x_dim, y_dim))
    y_mvn = random_mvn(batch_shape, y_dim)
    xy_mvn = random_mvn(batch_shape, x_dim + y_dim)
    gaussian = matrix_and_mvn_to_gaussian(matrix, y_mvn) + mvn_to_gaussian(xy_mvn)
    xy = torch.randn(sample_shape + (1,) * len(batch_shape) + (x_dim + y_dim,))
    x, y = xy[..., :x_dim], xy[..., x_dim:]
    y_pred = x.unsqueeze(-2).matmul(matrix).squeeze(-2)
    actual_log_prob = gaussian.log_density(xy)
    expected_log_prob = xy_mvn.log_prob(xy) + y_mvn.log_prob(y - y_pred)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("x_dim", [1, 2, 3])
@pytest.mark.parametrize("y_dim", [1, 2, 3])
def test_matrix_and_mvn_to_gaussian_2(sample_shape, batch_shape, x_dim, y_dim):
    matrix = torch.randn(batch_shape + (x_dim, y_dim))
    y_mvn = random_mvn(batch_shape, y_dim)
    x_mvn = random_mvn(batch_shape, x_dim)
    Mx_cov = matrix.transpose(-2, -1).matmul(x_mvn.covariance_matrix).matmul(matrix)
    Mx_loc = matrix.transpose(-2, -1).matmul(x_mvn.loc.unsqueeze(-1)).squeeze(-1)
    mvn = dist.MultivariateNormal(Mx_loc + y_mvn.loc, Mx_cov + y_mvn.covariance_matrix)
    expected = mvn_to_gaussian(mvn)

    actual = gaussian_tensordot(mvn_to_gaussian(x_mvn),
                                matrix_and_mvn_to_gaussian(matrix, y_mvn), dims=x_dim)
    assert_close_gaussian(expected, actual)


@pytest.mark.parametrize("x_batch_shape,y_batch_shape", [
    ((), ()),
    ((3,), ()),
    ((), (3,)),
    ((2, 1), (3,)),
    ((2, 3), (2, 3,)),
], ids=str)
@pytest.mark.parametrize("x_dim,y_dim,dot_dims", [
    (0, 0, 0),
    (0, 2, 0),
    (1, 0, 0),
    (2, 1, 0),
    (3, 3, 3),
    (3, 2, 1),
    (3, 2, 2),
    (5, 4, 2),
], ids=str)
@pytest.mark.parametrize("x_rank,y_rank", [
    (1, 1), (4, 1), (1, 4), (4, 4)
], ids=str)
def test_gaussian_tensordot(dot_dims,
                            x_batch_shape, x_dim, x_rank,
                            y_batch_shape, y_dim, y_rank):
    x_rank = min(x_rank, x_dim)
    y_rank = min(y_rank, y_dim)
    x = random_gaussian(x_batch_shape, x_dim, x_rank)
    y = random_gaussian(y_batch_shape, y_dim, y_rank)
    na = x_dim - dot_dims
    nb = dot_dims
    nc = y_dim - dot_dims
    try:
        torch.cholesky(x.precision[..., na:, na:] + y.precision[..., :nb, :nb])
    except RuntimeError:
        pytest.skip("Cannot marginalize the common variables of two Gaussians.")

    z = gaussian_tensordot(x, y, dot_dims)
    assert z.dim() == x_dim + y_dim - 2 * dot_dims

    # We make these precision matrices positive definite to test the math
    x.precision = x.precision + 1e-1 * torch.eye(x.dim())
    y.precision = y.precision + 1e-1 * torch.eye(y.dim())
    z = gaussian_tensordot(x, y, dot_dims)
    # compare against broadcasting, adding, and marginalizing
    precision = pad(x.precision, (0, nc, 0, nc)) + pad(y.precision, (na, 0, na, 0))
    info_vec = pad(x.info_vec, (0, nc)) + pad(y.info_vec, (na, 0))
    covariance = torch.inverse(precision)
    loc = covariance.matmul(info_vec.unsqueeze(-1)).squeeze(-1) if info_vec.size(-1) > 0 else info_vec
    z_covariance = torch.inverse(z.precision)
    z_loc = z_covariance.matmul(z.info_vec.view(z.info_vec.shape + (int(z.dim() > 0),))).sum(-1)
    assert_close(loc[..., :na], z_loc[..., :na])
    assert_close(loc[..., x_dim:], z_loc[..., na:])
    assert_close(covariance[..., :na, :na], z_covariance[..., :na, :na])
    assert_close(covariance[..., :na, x_dim:], z_covariance[..., :na, na:])
    assert_close(covariance[..., x_dim:, :na], z_covariance[..., na:, :na])
    assert_close(covariance[..., x_dim:, x_dim:], z_covariance[..., na:, na:])

    # Assume a = c = 0, integrate out b
    # FIXME: this might be not a stable way to compute integral
    num_samples = 200000
    scale = 20
    # generate samples in [-10, 10]
    value_b = torch.rand((num_samples,) + z.batch_shape + (nb,)) * scale - scale / 2
    value_x = pad(value_b, (na, 0))
    value_y = pad(value_b, (0, nc))
    expect = torch.logsumexp(x.log_density(value_x) + y.log_density(value_y), dim=0)
    expect += math.log(scale ** nb / num_samples)
    actual = z.log_density(torch.zeros(z.batch_shape + (z.dim(),)))
    # TODO(fehiepsi): find some condition to make this test stable, so we can compare large value
    # log densities.
    assert_close(actual.clamp(max=10.), expect.clamp(max=10.), atol=0.1, rtol=0.1)
