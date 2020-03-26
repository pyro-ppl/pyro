# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase

import pytest
import torch

import pyro.distributions as dist
import pyro.distributions.transforms as T

from functools import partial

pytestmark = pytest.mark.init(rng_seed=123)


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

    def _test_inverse(self, shape, transform):
        base_dist = dist.Normal(torch.zeros(shape), torch.ones(shape))

        x_true = base_dist.sample(torch.Size([10]))
        y = transform._call(x_true)

        # Cache is empty, hence must be calculating inverse afresh
        x_calculated = transform._inverse(y)

        assert torch.norm(x_true - x_calculated, dim=-1).max().item() < self.delta

    def _test_shape(self, base_shape, transform):
        base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        sample = dist.TransformedDistribution(base_dist, [transform]).sample()
        assert sample.shape == base_shape

    def _test(self, transform_factory, shape=True, jacobian=True, inverse=True):
        for input_dim in [2, 5, 10]:
            transform = transform_factory(input_dim)
            if jacobian:
                self._test_jacobian(input_dim, transform)
            if inverse:
                self._test_inverse(input_dim, transform)
            if shape:
                for shape in [(3,), (3, 4)]:
                    self._test_shape(shape + (input_dim,), transform)

    def _test_conditional(self, conditional_transform_factory, context_dim=3, **kwargs):
        def transform_factory(input_dim, context_dim=context_dim):
            z = torch.rand(1, context_dim)
            return conditional_transform_factory(input_dim, context_dim).condition(z)
        self._test(transform_factory, **kwargs)

    def test_affine_autoregressive(self):
        for stable in [True, False]:
            self._test(partial(T.affine_autoregressive, stable=stable))

    def test_affine_coupling(self):
        self._test(T.affine_coupling)

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

    def test_conditional_affine_coupling(self):
        self._test_conditional(T.conditional_affine_coupling)

    def test_conditional_planar(self):
        self._test_conditional(T.conditional_planar, inverse=False)

    def test_conditional_radial(self):
        self._test_conditional(T.conditional_radial, inverse=False)

    def test_discrete_cosine(self):
        # NOTE: Need following since helper function unimplemented
        self._test(lambda input_dim: T.DiscreteCosineTransform())

    def test_elu(self):
        # NOTE: Need following since helper function mistakenly doesn't take input dim
        self._test(lambda input_dim: T.elu())

    def test_generalized_channel_permute(self):
        for shape in [(3, 16, 16), (1, 3, 32, 32), (2, 5, 9, 64, 64)]:
            transform = T.generalized_channel_permute(channels=shape[-3])
            self._test_shape(shape, transform)
            self._test_inverse(shape, transform)

        for width_dim in [2, 4, 6]:
            # Do a bit of a hack until we merge in Reshape transform
            class Flatten(T.GeneralizedChannelPermute):
                event_dim = 1

                def _call(self, x):
                    return super(Flatten, self)._call(x.view(-1, 3, width_dim, width_dim)).view_as(x)

                def _inverse(self, x):
                    return super(Flatten, self)._inverse(x.view(-1, 3, width_dim, width_dim)).view_as(x)

            input_dim = (width_dim**2) * 3
            self._test_jacobian(input_dim, Flatten())

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
        self._test(T.permute)

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

    def test_tanh(self):
        # NOTE: Need following since helper function mistakenly doesn't take input dim
        self._test(lambda input_dim: T.tanh())
