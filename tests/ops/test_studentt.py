import math

import pytest
import torch
from torch.nn.functional import pad

import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.ops.studentt import (
    GaussianGamma,
    gaussian_gamma_tensordot,
    matrix_and_mvt_to_gaussian_gamma,
    mvt_to_gaussian_gamma,
)
from tests.common import assert_close
from tests.ops.studentt import assert_close_gaussian_gamma, random_gaussian_gamma, random_mvt


@pytest.mark.parametrize("extra_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("log_normalizer_shape,info_vec_shape,precision_shape,alpha_shape,beta_shape", [
    ((), (), (), (), ()),
    ((5,), (), (), (), ()),
    ((), (5,), (), (), ()),
    ((), (), (5,), (), ()),
    ((), (), (), (5,), ()),
    ((), (), (), (), (5,)),
    ((3, 1, 1), (1, 4, 1), (1, 1, 5), (3, 4, 1), (1, 4, 5)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_expand(extra_shape, log_normalizer_shape, info_vec_shape, precision_shape, alpha_shape, beta_shape, dim):
    rank = dim + dim
    log_normalizer = torch.randn(log_normalizer_shape)
    info_vec = torch.randn(info_vec_shape + (dim,))
    precision = torch.randn(precision_shape + (dim, rank))
    precision = precision.matmul(precision.transpose(-1, -2))
    alpha = torch.randn(alpha_shape).exp()
    beta = torch.randn(beta_shape).exp()
    gaussian_gamma = GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)

    expected_shape = extra_shape + broadcast_shape(
        log_normalizer_shape, info_vec_shape, precision_shape, alpha_shape, beta_shape)
    actual = gaussian_gamma.expand(expected_shape)
    assert actual.batch_shape == expected_shape


@pytest.mark.parametrize("old_shape,new_shape", [
    ((6,), (3, 2)),
    ((5, 6), (5, 3, 2)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_reshape(old_shape, new_shape, dim):
    gaussian_gamma = random_gaussian_gamma(old_shape, dim)

    # reshape to new
    new = gaussian_gamma.reshape(new_shape)
    assert new.batch_shape == new_shape

    # reshape back to old
    g = new.reshape(old_shape)
    assert_close_gaussian_gamma(g, gaussian_gamma)


@pytest.mark.parametrize("shape,cat_dim,split", [
    ((4, 7, 6), -1, (2, 1, 3)),
    ((4, 7, 6), -2, (1, 1, 2, 3)),
    ((4, 7, 6), 1, (1, 1, 2, 3)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_cat(shape, cat_dim, split, dim):
    assert sum(split) == shape[cat_dim]
    gaussian_gamma = random_gaussian_gamma(shape, dim)
    parts = []
    end = 0
    for size in split:
        beg, end = end, end + size
        if cat_dim == -1:
            part = gaussian_gamma[..., beg: end]
        elif cat_dim == -2:
            part = gaussian_gamma[..., beg: end, :]
        elif cat_dim == 1:
            part = gaussian_gamma[:, beg: end]
        else:
            raise ValueError
        parts.append(part)

    actual = GaussianGamma.cat(parts, cat_dim)
    assert_close_gaussian_gamma(actual, gaussian_gamma)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("left", [0, 1, 2])
@pytest.mark.parametrize("right", [0, 1, 2])
def test_pad(shape, left, right, dim):
    expected = random_gaussian_gamma(shape, dim)
    padded = expected.event_pad(left=left, right=right)
    assert padded.batch_shape == expected.batch_shape
    assert padded.dim() == left + expected.dim() + right
    mid = slice(left, padded.dim() - right)
    assert_close(padded.info_vec[..., mid], expected.info_vec)
    assert_close(padded.precision[..., mid, mid], expected.precision)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_add(shape, dim):
    x = random_gaussian_gamma(shape, dim)
    y = random_gaussian_gamma(shape, dim)
    value = torch.randn(dim)
    s = torch.randn(()).exp()
    assert_close((x + y).log_density(value, s), x.log_density(value, s) + y.log_density(value, s))


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize_shape(batch_shape, left, right):
    dim = left + right
    g = random_gaussian_gamma(batch_shape, dim)
    assert g.marginalize(left=left).dim() == right
    assert g.marginalize(right=right).dim() == left


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize(batch_shape, left, right):
    dim = left + right
    g = random_gaussian_gamma(batch_shape, dim)
    s = torch.randn(batch_shape).exp()
    assert_close(g.marginalize(left=left).event_logsumexp().log_density(s),
                 g.event_logsumexp().log_density(s))
    assert_close(g.marginalize(right=right).event_logsumexp().log_density(s),
                 g.event_logsumexp().log_density(s))


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize_condition(sample_shape, batch_shape, left, right):
    dim = left + right
    g = random_gaussian_gamma(batch_shape, dim)
    x = torch.randn(sample_shape + (1,) * len(batch_shape) + (right,))
    s = torch.randn(batch_shape).exp()
    assert_close(g.marginalize(left=left).log_density(x, s),
                 g.condition(x).event_logsumexp().log_density(s))


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_condition(sample_shape, batch_shape, left, right):
    dim = left + right
    g = random_gaussian_gamma(batch_shape, dim)
    g.precision += torch.eye(dim) * 0.1
    value = torch.randn(sample_shape + (1,) * len(batch_shape) + (dim,))
    left_value, right_value = value[..., :left], value[..., left:]

    conditioned = g.condition(right_value)
    assert conditioned.batch_shape == sample_shape + g.batch_shape
    assert conditioned.dim() == left

    s = torch.randn(batch_shape).exp()
    actual = conditioned.log_density(left_value, s)
    expected = g.log_density(value, s)
    assert_close(actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_logsumexp(batch_shape, dim):
    g = random_gaussian_gamma(batch_shape, dim)
    g.info_vec *= 0.1  # approximately centered
    g.precision += torch.eye(dim) * 0.1
    s = torch.randn(batch_shape).exp() + 0.2

    num_samples = 200000
    scale = 10
    samples = torch.rand((num_samples,) + (1,) * len(batch_shape) + (dim,)) * scale - scale / 2
    expected = g.log_density(samples, s).logsumexp(0) + math.log(scale ** dim / num_samples)
    actual = g.event_logsumexp().log_density(s)
    assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_mvt_to_gaussian_gamma(sample_shape, batch_shape, dim):
    mvt = random_mvt(batch_shape, dim)
    mvt.scale_tril = (mvt.covariance_matrix + 3 * torch.eye(dim)).cholesky()
    g = mvt_to_gaussian_gamma(mvt)
    value = mvt.sample(sample_shape)

    num_samples = 200000
    scale = 10
    s = torch.rand((num_samples,) + (1,) * len(sample_shape + batch_shape)) * scale
    # marginalize s
    actual_log_prob = g.log_density(value, s).logsumexp(0) + math.log(scale / num_samples)
    expected_log_prob = mvt.log_prob(value)
    assert_close(actual_log_prob, expected_log_prob, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_mvt_to_gaussian_gamma_return_conditional(sample_shape, batch_shape, dim):
    mvt = random_mvt(batch_shape, dim)
    mvt.scale_tril = (mvt.covariance_matrix + 7 * torch.eye(dim)).cholesky()
    g = mvt_to_gaussian_gamma(mvt, return_conditional=True)
    value = mvt.sample(sample_shape)
    s = torch.rand(sample_shape + batch_shape)

    actual_log_density = g.log_density(value, s)
    # p(x | s) = MVN(loc, precision=s * mvt.precision)
    gaussian = dist.MultivariateNormal(
        loc=mvt.loc,
        scale_tril=(mvt.scale_tril / s.sqrt().unsqueeze(-1).unsqueeze(-1)))
    expected_log_density = gaussian.log_prob(value)
    print(torch.max(((expected_log_density - actual_log_density) / actual_log_density).abs()))
    assert_close(actual_log_density, expected_log_density, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("x_dim", [1, 2, 3])
@pytest.mark.parametrize("y_dim", [1, 2, 3])
def test_matrix_and_mvt_to_gaussian_gamma(sample_shape, batch_shape, x_dim, y_dim):
    matrix = torch.randn(batch_shape + (x_dim, y_dim))
    y_mvt = random_mvt(batch_shape, y_dim)
    y_mvt.scale_tril = (y_mvt.covariance_matrix + 3 * torch.eye(y_dim)).cholesky()
    g = matrix_and_mvt_to_gaussian_gamma(matrix, y_mvt)
    xy = torch.randn(sample_shape + (1,) * len(batch_shape) + (x_dim + y_dim,))
    x, y = xy[..., :x_dim], xy[..., x_dim:]
    y_pred = x.unsqueeze(-2).matmul(matrix).squeeze(-2)

    num_samples = 200000
    scale = 10
    s = torch.rand((num_samples,) + (1,) * len(sample_shape + batch_shape)) * scale
    actual_log_prob = g.log_density(xy, s).logsumexp(0) + math.log(scale / num_samples)
    expected_log_prob = y_mvt.log_prob(y - y_pred)
    assert_close(actual_log_prob, expected_log_prob, atol=0.1, rtol=0.1)


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
def test_gaussian_gamma_tensordot(dot_dims,
                                  x_batch_shape, x_dim, x_rank,
                                  y_batch_shape, y_dim, y_rank):
    x_rank = min(x_rank, x_dim)
    y_rank = min(y_rank, y_dim)
    x = random_gaussian_gamma(x_batch_shape, x_dim, x_rank)
    y = random_gaussian_gamma(y_batch_shape, y_dim, y_rank)
    na = x_dim - dot_dims
    nb = dot_dims
    nc = y_dim - dot_dims
    try:
        torch.cholesky(x.precision[..., na:, na:] + y.precision[..., :nb, :nb])
    except RuntimeError:
        pytest.skip("Cannot marginalize the common variables of two Gaussians.")

    z = gaussian_gamma_tensordot(x, y, dot_dims)
    assert z.dim() == x_dim + y_dim - 2 * dot_dims

    # We make these precision matrices positive definite to test the math
    x.precision = x.precision + 3 * torch.eye(x.dim())
    y.precision = y.precision + 3 * torch.eye(y.dim())
    z = gaussian_gamma_tensordot(x, y, dot_dims)
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

    s = torch.randn(z.batch_shape).exp()
    # Assume a = c = 0, integrate out b
    num_samples = 200000
    scale = 10
    # generate samples in [-10, 10]
    value_b = torch.rand((num_samples,) + z.batch_shape + (nb,)) * scale - scale / 2
    value_x = pad(value_b, (na, 0))
    value_y = pad(value_b, (0, nc))
    expect = torch.logsumexp(x.log_density(value_x, s) + y.log_density(value_y, s), dim=0)
    expect += math.log(scale ** nb / num_samples)
    actual = z.log_density(torch.zeros(z.batch_shape + (z.dim(),)), s)
    assert_close(actual.clamp(max=10.), expect.clamp(max=10.), atol=0.1, rtol=0.1)
