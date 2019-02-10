from __future__ import absolute_import, division, print_function

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


@pytest.mark.parametrize("concentration_shape", [(), (1,), (3,), (5, 3)])
@pytest.mark.parametrize("sample_shape", [(), (1,), (3,), (5, 3)])
@pytest.mark.parametrize("sample_method", ["cvine", "onion"])
def test_shape(concentration_shape, sample_shape, sample_method):
    dimension = 5
    concentration = torch.rand(concentration_shape)
    d = LKJCorrCholesky(dimension, concentration, sample_method=sample_method)
    samples = d.sample(sample_shape)
    #log_prob = d.log_prob(samples)

    assert d.batch_shape == concentration_shape
    assert d.event_shape == torch.Size([dimension, dimension])
    assert samples.shape == sample_shape + d.batch_shape + d.event_shape
    #assert log_prob.shape == samples.shape[:-2]


@pytest.mark.parametrize("concentration", [0.5, 1, 2])
def test_sample(concentration):
    # We test the fact that marginal off-diagonal element of sampled correlation matrix is
    # Beta(concentration + (D - 2) / 2, concentration + (D - 2) / 2) on (-1, 1)
    dimension = 5
    d = LKJCorrCholesky(dimension, concentration, sample_method=sample_method)
    samples = d.sample(sample_shape=torch.Size([100000]))
    corr_samples = samples.matmul(samples.transpose(-1, -2))

    marginal = TransformedDistribution(
        Beta(concentration + (dimension - 2) / 2., concentration + (dimension - 2) / 2.),
        AffineTransform(loc=-1, scale=2))
    target_mean = samples.new_full((dimension, dimension), marginal.mean)
    target_mean[::dimension + 1] = 1  # diagonal elements of correlation matrices are 1
    target_variance = samples.new_full((dimension, dimension), marginal.variance)
    target_variance[::dimension + 1] = 0

    assert_tensors_equal(transform.domain.check(z),
                         x.new_ones(x_shape, dtype=torch.uint8))


@pytest.mark.parametrize("concentration", [0.5, 1, 2])
def test_log_prob("concentration", [0.5, 1, 2]):
    pass
