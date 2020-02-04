# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

import torch
from torch.distributions import Gamma, MultivariateNormal, StudentT

from pyro.distributions import MultivariateStudentT
from tests.common import assert_equal


def random_mvt(df_shape, loc_shape, cov_shape, dim):
    """
    Generate a random MultivariateStudentT distribution for testing.
    """
    rank = dim + dim
    df = torch.rand(df_shape, requires_grad=True).exp()
    loc = torch.randn(loc_shape + (dim,), requires_grad=True)
    cov = torch.randn(cov_shape + (dim, rank), requires_grad=True)
    cov = cov.matmul(cov.transpose(-1, -2))
    scale_tril = cov.cholesky()
    return MultivariateStudentT(df, loc, scale_tril)


@pytest.mark.parametrize('df_shape', [
    (), (2,), (3, 2),
])
@pytest.mark.parametrize('loc_shape', [
    (), (2,), (3, 2),
])
@pytest.mark.parametrize('cov_shape', [
    (), (2,), (3, 2),
])
@pytest.mark.parametrize('dim', [
    1, 3, 5,
])
def test_shape(df_shape, loc_shape, cov_shape, dim):
    mvt = random_mvt(df_shape, loc_shape, cov_shape, dim)
    assert mvt.df.shape == mvt.batch_shape
    assert mvt.loc.shape == mvt.batch_shape + mvt.event_shape
    assert mvt.covariance_matrix.shape == mvt.batch_shape + mvt.event_shape * 2
    assert mvt.scale_tril.shape == mvt.covariance_matrix.shape
    assert mvt.precision_matrix.shape == mvt.covariance_matrix.shape

    assert_equal(mvt.precision_matrix, mvt.covariance_matrix.inverse())

    # smoke test for precision/log_prob backward
    (mvt.precision_matrix.sum() + mvt.log_prob(torch.zeros(dim)).sum()).backward()


@pytest.mark.parametrize("batch_shape", [
    (),
    (3, 2),
    (4,),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2])
def test_log_prob(batch_shape, dim):
    loc = torch.randn(batch_shape + (dim,))
    A = torch.randn(batch_shape + (dim, dim + dim))
    scale_tril = A.matmul(A.transpose(-2, -1)).cholesky()
    x = torch.randn(batch_shape + (dim,))
    df = torch.randn(batch_shape).exp() + 2
    actual_log_prob = MultivariateStudentT(df, loc, scale_tril).log_prob(x)

    if dim == 1:
        expected_log_prob = StudentT(df.unsqueeze(-1), loc, scale_tril[..., 0]).log_prob(x).sum(-1)
        assert_equal(actual_log_prob, expected_log_prob)

    # test the fact MVT(df, loc, scale)(x) = int MVN(loc, scale / m)(x) Gamma(df/2,df/2)(m) dm
    num_samples = 100000
    gamma_samples = Gamma(df / 2, df / 2).sample(sample_shape=(num_samples,))
    mvn_scale_tril = scale_tril / gamma_samples.sqrt().unsqueeze(-1).unsqueeze(-1)
    mvn = MultivariateNormal(loc, scale_tril=mvn_scale_tril)
    expected_log_prob = mvn.log_prob(x).logsumexp(0) - math.log(num_samples)
    assert_equal(actual_log_prob, expected_log_prob, prec=0.01)


@pytest.mark.parametrize("df", [3.9, 9.1])
@pytest.mark.parametrize("dim", [1, 2])
def test_rsample(dim, df, num_samples=200 * 1000):
    scale_tril = (0.5 * torch.randn(dim)).exp().diag() + 0.1 * torch.randn(dim, dim)
    scale_tril = scale_tril.tril(0)
    scale_tril.requires_grad_(True)

    d = MultivariateStudentT(torch.tensor(df), torch.zeros(dim), scale_tril)
    z = d.rsample(sample_shape=(num_samples,))
    loss = z.pow(2.0).sum(-1).mean()
    loss.backward()

    actual_scale_tril_grad = scale_tril.grad.data.clone()
    scale_tril.grad.zero_()

    analytic = (df / (df - 2.0)) * torch.mm(scale_tril, scale_tril.t()).diag().sum()
    analytic.backward()
    expected_scale_tril_grad = scale_tril.grad.data

    assert_equal(expected_scale_tril_grad, actual_scale_tril_grad, prec=0.1)


@pytest.mark.parametrize("dim", [1, 2])
def test_log_prob_normalization(dim, df=6.1, grid_size=2000, domain_width=5.0):
    scale_tril = (0.2 * torch.randn(dim) - 1.5).exp().diag() + 0.1 * torch.randn(dim, dim)
    scale_tril = 0.1 * scale_tril.tril(0)

    volume_factor = domain_width
    prec = 0.01
    if dim == 2:
        volume_factor = volume_factor ** 2
        prec = 0.05

    sample_shape = (grid_size * grid_size, dim)
    z = torch.distributions.Uniform(-0.5 * domain_width, 0.5 * domain_width).sample(sample_shape)

    d = MultivariateStudentT(torch.tensor(df), torch.zeros(dim), scale_tril)
    normalizer = d.log_prob(z).exp().mean().item() * volume_factor

    assert_equal(normalizer, 1.0, prec=prec)


@pytest.mark.parametrize("batch_shape", [
    (),
    (3, 2),
    (4,),
], ids=str)
def test_mean_var(batch_shape):
    dim = 2
    loc = torch.randn(batch_shape + (dim,))
    A = torch.randn(batch_shape + (dim, dim + dim))
    scale_tril = A.matmul(A.transpose(-2, -1)).cholesky()
    df = torch.randn(batch_shape).exp() + 4
    num_samples = 100000
    d = MultivariateStudentT(df, loc, scale_tril)
    samples = d.sample(sample_shape=(num_samples,))
    expected_mean = samples.mean(0)
    expected_variance = samples.var(0)
    assert_equal(d.mean, expected_mean, prec=0.1)
    assert_equal(d.variance, expected_variance, prec=0.2)

    assert_equal(MultivariateStudentT(0.5, loc, scale_tril).mean,
                 torch.full(batch_shape + (dim,), float('nan')))
    assert_equal(MultivariateStudentT(0.5, loc, scale_tril).variance,
                 torch.full(batch_shape + (dim,), float('nan')))
    assert_equal(MultivariateStudentT(1.5, loc, scale_tril).variance,
                 torch.full(batch_shape + (dim,), float('inf')))
