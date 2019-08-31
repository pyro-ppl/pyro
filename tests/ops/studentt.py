import torch

import pyro.distributions as dist
from pyro.ops.studentt import GaussianGamma
from tests.common import assert_close


def random_gaussian_gamma(batch_shape, dim, rank=None):
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
    result = GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)
    assert result.dim() == dim
    assert result.batch_shape == batch_shape
    return result


def random_mvt(batch_shape, dim):
    """
    Generate a random MultivariateNormal distribution for testing.
    """
    rank = dim + dim
    df = torch.randn(batch_shape).exp() + 2
    loc = torch.randn(batch_shape + (dim,))
    cov = torch.randn(batch_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    scale_tril = cov.cholesky()
    return dist.MultivariateStudentT(df, loc, scale_tril)


def assert_close_gaussian_gamma(actual, expected):
    assert isinstance(actual, GaussianGamma)
    assert isinstance(expected, GaussianGamma)
    assert actual.dim() == expected.dim()
    assert actual.batch_shape == expected.batch_shape
    assert_close(actual.log_normalizer, expected.log_normalizer)
    assert_close(actual.info_vec, expected.info_vec)
    assert_close(actual.precision, expected.precision)
    assert_close(actual.alpha, expected.alpha)
    assert_close(actual.beta, expected.beta)
