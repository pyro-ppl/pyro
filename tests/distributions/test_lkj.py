from __future__ import absolute_import, division, print_function

import math

import pytest
import torch
from torch.distributions import biject_to, transform_to
from torch.distributions.transforms import AffineTransform

from pyro.distributions import Beta, TransformedDistribution
from pyro.distributions.lkj import corr_cholesky_constraint, LKJCorrCholesky
from tests.common import assert_tensors_equal, assert_close


@pytest.mark.parametrize("value_shape", [(1, 1), (3, 1, 1), (3, 3), (1, 3, 3), (5, 3, 3)])
def test_constraint(value_shape):
    value = torch.randn(value_shape).tril()
    value.diagonal(dim1=-2, dim2=-1).exp_()
    value = value / value.norm(2, dim=-1, keepdim=True)

    # this also tests for shape
    assert_tensors_equal(corr_cholesky_constraint.check(value),
                         value.new_ones(value_shape[:-2], dtype=torch.uint8))


def _autograd_log_det(ys, x):
    # computes log_abs_det_jacobian of y w.r.t. x
    return torch.stack([torch.autograd.grad(y, (x,), retain_graph=True)[0]
                        for y in ys]).slogdet()[1]


@pytest.mark.parametrize("x_shape", [(1,), (3, 1), (6,), (1, 6), (5, 6)])
@pytest.mark.parametrize("mapping", [biject_to, transform_to])
def test_corr_cholesky_transform(x_shape, mapping):
    transform = mapping(corr_cholesky_constraint)
    x = torch.randn(x_shape, requires_grad=True)
    y = transform(x)

    # test codomain
    assert_tensors_equal(transform.codomain.check(y),
                         x.new_ones(x_shape[:-1], dtype=torch.uint8))

    # test inv
    z = transform.inv(y)
    assert_close(x, z)

    # test domain
    assert_tensors_equal(transform.domain.check(z),
                         x.new_ones(x_shape, dtype=torch.uint8))

    # test log_abs_det_jacobian
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det.shape == x_shape[:-1]
    if len(x_shape) == 1:
        tril_index = y.new_ones(y.shape).tril(diagonal=-1) > 0.5
        y_tril_vector = y[tril_index]
        assert_close(_autograd_log_det(y_tril_vector, x), log_det)

        y_tril_vector = y_tril_vector.detach().requires_grad_()
        y = y.new_zeros(y.shape)
        y[tril_index] = y_tril_vector
        z = transform.inv(y)
        assert_close(_autograd_log_det(z, y_tril_vector), -log_det)


@pytest.mark.parametrize("concentration_shape", [(), (1,), (4,), (6, 4)])
@pytest.mark.parametrize("sample_shape", [(), (1,), (3,), (5, 3)])
@pytest.mark.parametrize("sample_method", ["cvine", "onion"])
def test_shape(concentration_shape, sample_shape, sample_method):
    dimension = 5
    concentration = torch.rand(concentration_shape)
    d = LKJCorrCholesky(dimension, concentration, sample_method=sample_method)
    samples = d.sample(sample_shape)
    log_prob = d.log_prob(samples)

    assert d.batch_shape == concentration_shape
    assert d.event_shape == torch.Size([dimension, dimension])
    assert samples.shape == sample_shape + d.batch_shape + d.event_shape
    assert log_prob.shape == samples.shape[:-2]


@pytest.mark.parametrize("concentration", [0.5, 1, 2])
@pytest.mark.parametrize("sample_method", ["cvine", "onion"])
def test_sample(concentration, sample_method):
    # We test for the fact that the marginal distribution of off-diagonal elements of sampled
    # correlation matrices is Beta(concentration + (D - 2) / 2, concentration + (D - 2) / 2)
    # scaled by a linear mapping X -> 2 * X - 1 (to make its support on (-1, 1))
    dimension = 5
    d = LKJCorrCholesky(dimension, concentration, sample_method=sample_method)
    samples = d.sample(sample_shape=torch.Size([50000]))
    corr_samples = samples.matmul(samples.transpose(-1, -2))

    marginal = Beta(concentration + (dimension - 2) / 2.,
                    concentration + (dimension - 2) / 2.)
    marginal_mean = 2 * marginal.mean - 1
    marginal_variance = 4 * marginal.variance
    target_mean = samples.new_full((dimension, dimension), marginal_mean)
    # diagonal elements of correlation matrices are 1
    target_mean.view(-1)[::dimension + 1] = 1
    # we multiply std by 2 because the marginal support is (-1, 1), not (0, 1)
    target_variance = samples.new_full((dimension, dimension), marginal_variance)
    target_variance.view(-1)[::dimension + 1] = 0

    assert_close(corr_samples.mean(dim=0), target_mean, atol=0.005)
    assert_close(corr_samples.var(dim=0), target_variance, atol=0.005)


@pytest.mark.parametrize("dimension", [2, 5, 10])
def test_log_prob_uniform(dimension):
    d = LKJCorrCholesky(dimension, concentration=1)
    sample = d.sample()

    # when concentration = 1, LKJ gives a uniform distribution over correlation matrix,
    # hence density of a correlation matrix will be Uniform((-1, 1)^D) = 0.5^D
    uniform_log_prob = -dimension * math.log(2)

    # we need to compute jacobian from cholesky -> corr
    tril_index = sample.new_ones(dimension, dimension).tril(diagonal=-1) > 0.5
    sample_tril = sample[tril_index].clone().detach().requires_grad_()
    sample_cloned = sample_tril.new_ones(dimension, dimension)
    sample_cloned[tril_index] = sample_tril
    sample_cloned.view(-1)[::dimension + 1] = (1 - sample_cloned.pow(2).sum(-1)).sqrt()
    corr = sample_cloned.matmul(sample_cloned.t())
    corr_tril = corr[tril_index]

    cholesky_to_corr_jacobian = _autograd_log_det(corr_tril, sample_tril)
    target_log_prob = cholesky_to_corr_jacobian + uniform_log_prob

    assert_close(d.log_prob(sample), target_log_prob)


@pytest.mark.parametrize("concentration", [0.5, 2])
def test_log_prob(concentration):
    if concentration == 1:
        pass
        # test for the 
