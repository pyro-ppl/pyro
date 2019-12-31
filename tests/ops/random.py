import torch

import pyro.distributions as dist
from pyro.ops.gamma_gaussian import GammaGaussian
from pyro.ops.gaussian import Gaussian
from pyro.ops.studentt import StudentT
from tests.common import assert_close


def random_gaussian(batch_shape, dim, rank=None):
    """
    Generate a random Gaussian for testing.
    """
    if rank is None:
        rank = dim + dim
    log_normalizer = torch.randn(batch_shape)
    info_vec = torch.randn(batch_shape + (dim,))
    samples = torch.randn(batch_shape + (dim, rank))
    precision = torch.matmul(samples, samples.transpose(-2, -1))
    result = Gaussian(log_normalizer, info_vec, precision)
    assert result.dim() == dim
    assert result.batch_shape == batch_shape
    return result


def random_mvn(batch_shape, dim):
    """
    Generate a random MultivariateNormal distribution for testing.
    """
    rank = dim + dim
    loc = torch.randn(batch_shape + (dim,))
    cov = torch.randn(batch_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    return dist.MultivariateNormal(loc, cov)


def assert_close_gaussian(actual, expected):
    assert isinstance(actual, Gaussian)
    assert isinstance(expected, Gaussian)
    assert actual.dim() == expected.dim()
    assert actual.batch_shape == expected.batch_shape
    assert_close(actual.log_normalizer, expected.log_normalizer)
    assert_close(actual.info_vec, expected.info_vec)
    assert_close(actual.precision, expected.precision)


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


def random_studentt(batch_shape, dim, rank=None):
    """
    Generate a random StudentT for testing.
    """
    if rank is None:
        rank = dim + dim
    log_normalizer = torch.randn(batch_shape)
    info_vec = torch.randn(batch_shape + (dim,))
    samples = torch.randn(batch_shape + (dim, rank))
    precision = torch.matmul(samples, samples.transpose(-2, -1))
    df = torch.randn(batch_shape).exp()
    rank = df.new_full(batch_shape, dim)
    result = StudentT(log_normalizer, info_vec, precision, df, rank)
    assert result.dim() == dim
    assert result.batch_shape == batch_shape
    return result


def random_mvt(batch_shape, dim):
    """
    Generate a random MultivariateNormal distribution for testing.
    """
    rank = dim + dim
    loc = torch.randn(batch_shape + (dim,))
    cov = torch.randn(batch_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    df = torch.randn(batch_shape).exp()
    return dist.MultivariateStudentT(df, loc, cov.cholesky())


def assert_close_studentt(actual, expected):
    assert isinstance(actual, StudentT)
    assert isinstance(expected, StudentT)
    assert actual.dim() == expected.dim()
    assert actual.batch_shape == expected.batch_shape
    assert_close(actual.log_normalizer, expected.log_normalizer)
    assert_close(actual.info_vec, expected.info_vec)
    assert_close(actual.precision, expected.precision)
    assert_close(actual.df, expected.df)
    assert_close(actual.rank, expected.rank)
