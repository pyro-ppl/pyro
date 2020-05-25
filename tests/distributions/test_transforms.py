# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase

import pytest
import torch

import pyro.distributions as dist
import pyro.distributions.transforms as T

from functools import partial, reduce
import operator

pytestmark = pytest.mark.init(rng_seed=123)


class Flatten(dist.TransformModule):
    """
    Used to handle transforms with `event_dim > 1` until we have a Reshape transform in PyTorch
    """
    event_dim = 1

    def __init__(self, transform, input_shape):
        super().__init__(cache_size=1)
        assert(transform.event_dim == len(input_shape))

        self.transform = transform
        self.input_shape = input_shape

    def _unflatten(self, x):
        return x.view(x.shape[:-1] + self.input_shape)

    def _call(self, x):
        return self.transform._call(self._unflatten(x)).view_as(x)

    def _inverse(self, x):
        return self.transform._inverse(self._unflatten(x)).view_as(x)

    def log_abs_det_jacobian(self, x, y):
        return self.transform.log_abs_det_jacobian(self._unflatten(x), self._unflatten(y))


class TransformTests(TestCase):
    def setUp(self):
        # Epsilon is used to compare numerical gradient to analytical one
        self.epsilon = 1e-4

        # Delta is tolerance for testing inverse, f(f^{-1}(x)) = x
        self.delta = 1e-6

    def _test_jacobian(self, input_dim, transform):
        jacobian = torch.zeros(input_dim, input_dim)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        x = torch.randn(1, input_dim)
        y = transform(x)
        if transform.event_dim == 1:
            analytic_ldt = transform.log_abs_det_jacobian(x, y).data
        else:
            analytic_ldt = transform.log_abs_det_jacobian(x, y).sum(-1).data

        for j in range(input_dim):
            for k in range(input_dim):
                epsilon_vector = torch.zeros(1, input_dim)
                epsilon_vector[0, j] = self.epsilon
                delta = (transform(x + 0.5 * epsilon_vector) - transform(x - 0.5 * epsilon_vector)) / self.epsilon
                jacobian[j, k] = float(delta[0, k].data.sum())

        # Apply permutation for autoregressive flows with a network
        if hasattr(transform, 'arn') and 'get_permutation' in dir(transform.arn):
            permutation = transform.arn.get_permutation()
            permuted_jacobian = jacobian.clone()
            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]
            jacobian = permuted_jacobian

        # For autoregressive flow, Jacobian is sum of diagonal, otherwise need full determinate
        if hasattr(transform, 'autoregressive') and transform.autoregressive:
            numeric_ldt = torch.sum(torch.log(torch.diag(jacobian)))
        else:
            numeric_ldt = torch.log(torch.abs(jacobian.det()))

        ldt_discrepancy = (analytic_ldt - numeric_ldt).abs()
        assert ldt_discrepancy < self.epsilon

        # Test that lower triangular with unit diagonal for autoregressive flows
        if hasattr(transform, 'autoregressive'):
            diag_sum = torch.sum(torch.diag(nonzero(jacobian)))
            lower_sum = torch.sum(torch.tril(nonzero(jacobian), diagonal=-1))
            assert diag_sum == float(input_dim)
            assert lower_sum == float(0.0)

    def _test_inverse(self, shape, transform):
        # Test g^{-1}(g(x)) = x
        # NOTE: Calling _call and _inverse directly bypasses caching
        base_dist = dist.Normal(torch.zeros(shape), torch.ones(shape))
        x_true = base_dist.sample(torch.Size([10]))
        y = transform._call(x_true)
        J_1 = transform.log_abs_det_jacobian(x_true, y)
        x_calculated = transform._inverse(y)
        J_2 = transform.log_abs_det_jacobian(x_true, y)
        assert (x_true - x_calculated).abs().max().item() < self.delta

        # Test that Jacobian after inverse op is same as after forward
        assert (J_1 - J_2).abs().max().item() < self.delta

    def _test_shape(self, base_shape, transform):
        base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        sample = dist.TransformedDistribution(base_dist, [transform]).sample()
        assert sample.shape == base_shape

    def _test(self, transform_factory, shape=True, jacobian=True, inverse=True, event_dim=1):
        for event_shape in [(2,), (5,)]:
            if event_dim > 1:
                event_shape = tuple([event_shape[0] + i for i in range(event_dim)])
            transform = transform_factory(event_shape[0] if len(event_shape) == 1 else event_shape)

            if inverse:
                self._test_inverse(event_shape, transform)
            if shape:
                for shape in [(3,), (3, 4), (3, 4, 5)]:
                    base_shape = shape + event_shape
                    self._test_shape(base_shape, transform)
            if jacobian:
                if event_dim > 1:
                    transform = Flatten(transform, event_shape)
                self._test_jacobian(reduce(operator.mul, event_shape, 1), transform)

    def _test_conditional(self, conditional_transform_factory, context_dim=3, event_dim=1, **kwargs):
        def transform_factory(input_dim, context_dim=context_dim):
            z = torch.rand(1, context_dim)
            return conditional_transform_factory(input_dim, context_dim).condition(z)
        self._test(transform_factory, event_dim=event_dim, **kwargs)

    def test_affine_autoregressive(self):
        for stable in [True, False]:
            self._test(partial(T.affine_autoregressive, stable=stable))

    def test_affine_coupling(self):
        for dim in [-1, -2]:
            self._test(partial(T.affine_coupling, dim=dim), event_dim=-dim)

    def test_batchnorm(self):
        # Need to make moving average statistics non-zeros/ones and set to eval so inverse is valid
        # (see the docs about the differing behaviour of BatchNorm in train and eval modes)
        def transform_factory(input_dim):
            transform = T.batchnorm(input_dim)
            transform._inverse(torch.normal(torch.arange(0., input_dim), torch.arange(1., 1. + input_dim) / input_dim))
            transform.eval()
            return transform

        self._test(transform_factory)

    def test_block_autoregressive_jacobians(self):
        for activation in ['ELU', 'LeakyReLU', 'sigmoid', 'tanh']:
            self._test(partial(T.block_autoregressive, activation=activation), inverse=False)

        for residual in [None, 'normal', 'gated']:
            self._test(partial(T.block_autoregressive, residual=residual), inverse=False)

    def test_conditional_affine_autoregressive(self):
        self._test_conditional(T.conditional_affine_autoregressive)

    def test_conditional_affine_coupling(self):
        for dim in [-1, -2]:
            self._test_conditional(partial(T.conditional_affine_coupling, dim=dim), event_dim=-dim)

    def test_conditional_generalized_channel_permute(self, context_dim=3):
        for shape in [(3, 16, 16), (1, 3, 32, 32), (2, 5, 3, 64, 64)]:
            # NOTE: Without changing the interface to generalized_channel_permute I can't reuse general
            # test for `event_dim > 1` transforms
            z = torch.rand(context_dim)
            transform = T.conditional_generalized_channel_permute(context_dim=3, channels=shape[-3]).condition(z)
            self._test_shape(shape, transform)
            self._test_inverse(shape, transform)

        for width_dim in [2, 4, 6]:
            input_dim = (width_dim**2) * 3
            self._test_jacobian(input_dim, Flatten(transform, (3, width_dim, width_dim)))

    def test_conditional_householder(self):
        self._test_conditional(T.conditional_householder)
        self._test_conditional(partial(T.conditional_householder, count_transforms=2))

    def test_conditional_neural_autoregressive(self):
        self._test_conditional(T.conditional_neural_autoregressive, inverse=False)

    def test_conditional_planar(self):
        self._test_conditional(T.conditional_planar, inverse=False)

    def test_conditional_radial(self):
        self._test_conditional(T.conditional_radial, inverse=False)

    def test_discrete_cosine(self):
        # NOTE: Need following since helper function unimplemented
        self._test(lambda input_dim: T.DiscreteCosineTransform())
        self._test(lambda input_dim: T.DiscreteCosineTransform(smooth=0.5))
        self._test(lambda input_dim: T.DiscreteCosineTransform(smooth=1.0))
        self._test(lambda input_dim: T.DiscreteCosineTransform(smooth=2.0))

    def test_haar_transform(self):
        # NOTE: Need following since helper function unimplemented
        self._test(lambda input_dim: T.HaarTransform(flip=True))
        self._test(lambda input_dim: T.HaarTransform(flip=False))

    def test_elu(self):
        # NOTE: Need following since helper function mistakenly doesn't take input dim
        self._test(lambda input_dim: T.elu())

    def test_generalized_channel_permute(self):
        for shape in [(3, 16, 16), (1, 3, 32, 32), (2, 5, 3, 64, 64)]:
            # NOTE: Without changing the interface to generalized_channel_permute I can't reuse general
            # test for `event_dim > 1` transforms
            transform = T.generalized_channel_permute(channels=shape[-3])
            self._test_shape(shape, transform)
            self._test_inverse(shape, transform)

        for width_dim in [2, 4, 6]:
            input_dim = (width_dim**2) * 3
            self._test_jacobian(input_dim, Flatten(transform, (3, width_dim, width_dim)))

    def test_householder(self):
        self._test(partial(T.householder, count_transforms=2))

    def test_leaky_relu(self):
        # NOTE: Need following since helper function mistakenly doesn't take input dim
        self._test(lambda input_dim: T.leaky_relu())

    def test_lower_cholesky_affine(self):
        # NOTE: Need following since helper function unimplemented
        def transform_factory(input_dim):
            loc = torch.randn(input_dim)
            scale_tril = torch.randn(input_dim).exp().diag() + 0.03 * torch.randn(input_dim, input_dim)
            scale_tril = scale_tril.tril(0)
            return T.LowerCholeskyAffine(loc, scale_tril)

        self._test(transform_factory)

    def test_neural_autoregressive(self):
        for activation in ['ELU', 'LeakyReLU', 'sigmoid', 'tanh']:
            self._test(partial(T.neural_autoregressive, activation=activation), inverse=False)

    def test_permute(self):
        for dim in [-1, -2]:
            self._test(partial(T.permute, dim=dim), event_dim=-dim)

    def test_planar(self):
        self._test(T.planar, inverse=False)

    def test_polynomial(self):
        self._test(T.polynomial, inverse=False)

    def test_radial(self):
        self._test(T.radial, inverse=False)

    def test_spline(self):
        self._test(T.spline)

    def test_sylvester(self):
        self._test(T.sylvester, inverse=False)
