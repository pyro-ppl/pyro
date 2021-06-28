import math

import pytest
import torch
from torch.nn.functional import pad

import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.ops.gaussian_sqrt import (
    AffineNormalS,
    GaussianS,
    gaussian_tensordotS,
    matrix_and_mvn_to_gaussianS,
    mvn_to_gaussianS,
)
from tests.common import assert_close
from tests.ops.gaussian import assert_close_gaussian, random_gaussian, random_gaussianS, random_mvn


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
    prec_sqrt = torch.randn(precision_shape + (dim, rank))
    gaussian = GaussianS(log_normalizer, info_vec, prec_sqrt)

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
    gaussian = random_gaussianS(old_shape, dim)

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
    gaussian = random_gaussianS(shape, dim)
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

    actual = GaussianS.cat(parts, cat_dim)
    assert_close_gaussian(actual, gaussian)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("left", [0, 1, 2])
@pytest.mark.parametrize("right", [0, 1, 2])
def test_pad(shape, left, right, dim):
    expected = random_gaussianS(shape, dim)
    padded = expected.event_pad(left=left, right=right)
    assert padded.batch_shape == expected.batch_shape
    assert padded.dim() == left + expected.dim() + right
    mid = slice(left, padded.dim() - right)
    assert_close(padded.info_vec[..., mid], expected.info_vec)
    assert_close(padded.precision[..., mid, mid], expected.precision)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_add(shape, dim):
    x = random_gaussianS(shape, dim)
    y = random_gaussianS(shape, dim)
    value = torch.randn(dim)
    print(x.prec_sqrt.shape, y.prec_sqrt.shape, (x + y).prec_sqrt.shape)
    assert_close((x + y).log_density(value), x.log_density(value) + y.log_density(value))


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize_shape(batch_shape, left, right):
    dim = left + right
    g = random_gaussianS(batch_shape, dim)
    assert g.marginalize(left=left).dim() == right
    assert g.marginalize(right=right).dim() == left


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize(batch_shape, left, right):
    dim = left + right
    g = random_gaussianS(batch_shape, dim)
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
    g = random_gaussianS(batch_shape, dim)
    x = torch.randn(sample_shape + (1,) * len(batch_shape) + (right,))
    assert_close(g.marginalize(left=left).log_density(x),
                 g.condition(x).event_logsumexp())


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_condition(sample_shape, batch_shape, left, right):
    dim = left + right
    gaussian = random_gaussianS(batch_shape, dim)
    value = torch.randn(sample_shape + (1,) * len(batch_shape) + (dim,))
    left_value, right_value = value[..., :left], value[..., left:]

    conditioned = gaussian.condition(right_value)
    assert conditioned.batch_shape == sample_shape + gaussian.batch_shape
    assert conditioned.dim() == left

    actual = conditioned.log_density(left_value)
    expected = gaussian.log_density(value)
    assert_close(actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_logsumexp(batch_shape, dim):
    gaussian = random_gaussian(batch_shape, dim)
    gaussian.info_vec *= 0.1  # approximately centered
    gaussian.precision += torch.eye(dim) * 0.1
    gaussian.prec_sqrt = torch.cholesky(gaussian.precision + torch.eye(dim) * 0.1)

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
    actual = matrix_and_mvn_to_gaussianS(matrix, normal)
    assert isinstance(actual, AffineNormalS)
    actual_like = actual.condition(y)
    assert isinstance(actual_like, GaussianS)

    mvn = dist.MultivariateNormal(loc, scale_tril=scale.diag_embed())
    expected = matrix_and_mvn_to_gaussianS(matrix, mvn)
    assert isinstance(expected, GaussianS)
    expected_like = expected.condition(y)
    assert isinstance(expected_like, GaussianS)

    assert_close(actual_like.log_normalizer, expected_like.log_normalizer)
    assert_close(actual_like.info_vec, expected_like.info_vec)
    assert_close(actual_like.precision, expected_like.precision)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_mvn_to_gaussian(sample_shape, batch_shape, dim):
    mvn = random_mvn(batch_shape, dim)
    gaussian = mvn_to_gaussianS(mvn)
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
    gaussian = matrix_and_mvn_to_gaussianS(matrix, y_mvn) + mvn_to_gaussianS(xy_mvn)
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
    expected = mvn_to_gaussianS(mvn)

    actual = gaussian_tensordotS(mvn_to_gaussianS(x_mvn),
                                 matrix_and_mvn_to_gaussianS(matrix, y_mvn), dims=x_dim)
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
    x = random_gaussianS(x_batch_shape, x_dim, x_rank)
    y = random_gaussianS(y_batch_shape, y_dim, y_rank)
    na = x_dim - dot_dims
    nb = dot_dims
    nc = y_dim - dot_dims

    skip = False
    try:
        torch.cholesky(x.precision[..., na:, na:] + y.precision[..., :nb, :nb])
    except RuntimeError:
        skip = True
    if skip or (x_rank + y_rank) < dot_dims:
        pytest.skip("Cannot marginalize the common variables of two Gaussians.")

    z = gaussian_tensordotS(x, y, dot_dims)
    assert z.dim() == x_dim + y_dim - 2 * dot_dims

    # We make these precision matrices positive definite to test the math
    x.precision = x.precision + 1e-1 * torch.eye(x.dim())
    x.prec_sqrt = x.precision.cholesky()
    y.precision = y.precision + 1e-1 * torch.eye(y.dim())
    y.prec_sqrt = y.precision.cholesky()
    z = gaussian_tensordotS(x, y, dot_dims)
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

    # test triangularize
    assert z.rank() == x.rank() + y.rank()
    z_tri = z.triangularize()
    assert z_tri.rank() == min(z.rank(), z.dim())
    assert_close(z_tri.precision, z.precision)

    # Assume a = c = 0, integrate out b
    num_samples = 200000
    scale = 20
    # generate samples in [-10, 10]
    value_b = torch.rand((num_samples,) + z.batch_shape + (nb,)) * scale - scale / 2
    value_x = pad(value_b, (na, 0))
    value_y = pad(value_b, (0, nc))
    expect = torch.logsumexp(x.log_density(value_x) + y.log_density(value_y), dim=0)
    expect += math.log(scale ** nb / num_samples)
    actual = z.log_density(torch.zeros(z.batch_shape + (z.dim(),)))
    assert_close(actual.clamp(max=10.), expect.clamp(max=10.), atol=0.1, rtol=0.1)
