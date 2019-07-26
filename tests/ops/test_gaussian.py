import math

import pytest
import torch
from torch.nn.functional import pad

from pyro.ops.gaussian import Gaussian, gaussian_tensordot
from tests.common import assert_close


def random_gaussian(batch_shape, dim, rank):
    """
    Generate a random Gaussian for testing.
    """
    log_normalizer = torch.randn(batch_shape)
    info_vec = torch.randn(batch_shape + (dim,))
    rank = min(dim, rank)
    samples = torch.randn(batch_shape + (dim, rank))
    precision = torch.matmul(samples, samples.transpose(-2, -1))
    result = Gaussian(log_normalizer, info_vec, precision)
    assert result.dim() == dim
    assert result.batch_shape == batch_shape
    return result


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
