# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase

import pytest
import torch

import pyro.distributions as dist
import pyro.distributions.transforms as T

pytestmark = pytest.mark.init(rng_seed=123)


class TransformTests(TestCase):
    def setUp(self):
        # Epsilon is used to compare numerical gradient to analytical one
        self.epsilon = 1e-4

        # Delta is tolerance for testing f(f^{-1}(x)) = x
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
        if hasattr(transform, 'arn'):
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
        if hasattr(transform, 'arn'):
            diag_sum = torch.sum(torch.diag(nonzero(jacobian)))
            lower_sum = torch.sum(torch.tril(nonzero(jacobian), diagonal=-1))
            assert diag_sum == float(input_dim)
            assert lower_sum == float(0.0)

    def _test_inverse(self, input_dim, transform):
        base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))

        x_true = base_dist.sample(torch.Size([10]))
        y = transform._call(x_true)

        # Cache is empty, hence must be calculating inverse afresh
        x_calculated = transform._inverse(y)

        assert torch.norm(x_true - x_calculated, dim=-1).max().item() < self.delta

    def _test_shape(self, base_shape, transform):
        base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        sample = dist.TransformedDistribution(base_dist, [transform]).sample()
        assert sample.shape == base_shape

    # Affine autoregressive
    def test_affine_autoregressive(self):
        for stable in [True, False]:
            for input_dim in [2, 5, 10]:
                transform = T.affine_autoregressive(input_dim, stable=stable)
                self._test_jacobian(input_dim, transform)
                self._test_inverse(input_dim, transform)
                for shape in [(3,), (3, 4)]:
                    self._test_shape(shape + (input_dim,), transform)
                
    # Affine Coupling
    def test_affine_coupling(self):
        for input_dim in [2, 5, 10]:
            transform = T.affine_coupling(input_dim)
            self._test_inverse(input_dim, transform)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Batchnorm
    def test_batchnorm(self):
        for input_dim in [2, 5, 10]:
            transform = T.batchnorm(input_dim)
            transform._inverse(torch.normal(torch.arange(0., input_dim), torch.arange(1., 1. + input_dim) / input_dim))
            transform.eval()
            self._test_jacobian(input_dim, transform)
            self._test_inverse(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)            

    # Block autoregresive
    def test_block_autoregressive_jacobians(self):
        for activation in ['ELU', 'LeakyReLU', 'sigmoid', 'tanh']:
            for input_dim in [2, 5, 10]:
                transform = T.block_autoregressive(input_dim, activation=activation)
                self._test_jacobian(input_dim, transform)
                for shape in [(3,), (3, 4)]:
                    self._test_shape(shape + (input_dim,), transform)

        for residual in [None, 'normal', 'gated']:
            for input_dim in [2, 5, 10]:
                transform = T.block_autoregressive(input_dim, residual=residual)
                self._test_jacobian(input_dim, transform)
                for shape in [(3,), (3, 4)]:
                    self._test_shape(shape + (input_dim,), transform)

    # Conditional planar
    def test_conditional_planar(self):
        observed_dim = 3
        for input_dim in [2, 5, 10]:
            z = torch.rand(observed_dim)
            transform = T.conditional_planar(input_dim, observed_dim).condition(z)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Discrete Cosine Transform
    def test_discrete_cosine(self):
        for input_dim in [2, 5, 10]:
            transform = T.DiscreteCosineTransform()
            self._test_jacobian(input_dim, transform)
            self._test_inverse(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)        

    # ELU
    def test_elu(self):
        for input_dim in [2, 5, 10]:
            transform = T.elu()
            self._test_inverse(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Householder
    def test_householder(self):
        for input_dim in [2, 5, 10]:
            transform = T.householder(input_dim, count_transforms=2)
            self._test_inverse(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Leaky ReLU
    def test_leaky_relu(self):
        for input_dim in [2, 5, 10]:
            transform = T.leaky_relu()
            self._test_inverse(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)        

    # Lower Cholesky affine
    # TODO: Create a affine_lower_cholesky helper function to simplify the following
    def test_lower_cholesky_affine(self):
        for input_dim in [2, 3]:
            loc = torch.randn(input_dim)
            scale_tril = torch.randn(input_dim).exp().diag() + 0.03 * torch.randn(input_dim, input_dim)
            scale_tril = scale_tril.tril(0)
            transform = T.LowerCholeskyAffine(loc, scale_tril)
            self._test_inverse(input_dim, transform)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Neural autoregressive
    def test_neural_autoregressive(self):
        for activation in ['ELU', 'LeakyReLU', 'sigmoid', 'tanh']:
            for input_dim in [2, 5, 10]:
                transform = T.neural_autoregressive(input_dim, activation=activation)
                self._test_jacobian(input_dim, transform)
                for shape in [(3,), (3, 4)]:
                    self._test_shape(shape + (input_dim,), transform)
    # Permute
    def test_permute(self):
        for input_dim in [2, 5, 10]:
            transform = T.permute(input_dim)
            self._test_inverse(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                    self._test_shape(shape + (input_dim,), transform)

    # Planar flow
    def test_planar(self):
        for input_dim in [2, 5, 10]:
            transform = T.planar(input_dim)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Polynomial
    def test_polynomial(self):
        for input_dim in [2, 5, 10]:
            transform = T.polynomial(input_dim)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)
    
    # Radial
    def test_radial(self):
        for input_dim in [2, 5, 10]:
            transform = T.radial(input_dim)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Sylvester flow
    def test_sylvester(self):
        for input_dim in [2, 5, 10]:
            transform = T.sylvester(input_dim)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)

    # Tanh
    def test_tanh(self):
        for input_dim in [2, 5, 10]:
            transform = T.tanh()
            self._test_inverse(input_dim, transform)
            self._test_jacobian(input_dim, transform)
            for shape in [(3,), (3, 4)]:
                self._test_shape(shape + (input_dim,), transform)
