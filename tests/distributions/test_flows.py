from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np
import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.iaf import InverseAutoregressiveFlow, InverseAutoregressiveFlowStable
from pyro.nn import AutoRegressiveNN

pytestmark = pytest.mark.init(rng_seed=123)


class AutoregressiveFlowTests(TestCase):
    def setUp(self):
        # Epsilon is used to compare numerical gradient to analytical one
        # NOTE: This is required to be larger than delta due to shortcomings in first order finite-difference methods
        # TODO: Investigate better methods for calculating numerical gradients
        self.epsilon = 1.0e-3

        # Delta is used to compare numerical inverse to analytical one
        self.delta = 1.0e-6

    def _test_jacobian(self, input_dim, make_flow):
        jacobian = torch.zeros(input_dim, input_dim)
        iaf = make_flow(input_dim)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        x = torch.randn(1, input_dim)
        iaf_x = iaf(x)
        analytic_ldt = iaf.log_abs_det_jacobian(x, iaf_x).data.sum()

        for j in range(input_dim):
            for k in range(input_dim):
                epsilon_vector = torch.zeros(1, input_dim)
                epsilon_vector[0, j] = self.epsilon
                delta = (iaf(x + 0.5 * epsilon_vector) - iaf(x - 0.5 * epsilon_vector)) / self.epsilon
                jacobian[j, k] = float(delta[0, k].data.sum())

        permutation = iaf.arn.get_permutation()
        permuted_jacobian = jacobian.clone()
        for j in range(input_dim):
            for k in range(input_dim):
                permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]
        numeric_ldt = torch.sum(torch.log(torch.diag(permuted_jacobian)))
        ldt_discrepancy = np.fabs(analytic_ldt - numeric_ldt)

        diag_sum = torch.sum(torch.diag(nonzero(permuted_jacobian)))
        lower_sum = torch.sum(torch.tril(nonzero(permuted_jacobian), diagonal=-1))

        assert ldt_discrepancy < self.epsilon
        assert diag_sum == float(input_dim)
        assert lower_sum == float(0.0)

    def _test_inverse(self, input_dim, make_flow):
        base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
        iaf = make_flow(input_dim)
        iaf_dist = dist.TransformedDistribution(base_dist, [iaf])

        # The first time _inverse is called it returns the cached value, the second time the numerical inverse
        y = iaf_dist.sample(torch.Size([100]))
        x_true = iaf._inverse(y)
        x_calculated = iaf._inverse(y)

        assert torch.norm(x_true - x_calculated, dim=-1).max().item() < self.delta

    def _test_shape(self, base_shape, make_flow):
        base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        last_dim = base_shape[-1] if isinstance(base_shape, tuple) else base_shape
        iaf = make_flow(input_dim=last_dim)
        sample = dist.TransformedDistribution(base_dist, [iaf]).sample()
        assert sample.shape == base_shape

    def _make_iaf(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3*input_dim + 1])
        return InverseAutoregressiveFlow(arn)

    def _make_iaf_stable(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3*input_dim + 1])
        return InverseAutoregressiveFlowStable(arn, sigmoid_bias=0.5)

    def test_iaf_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_iaf)

    def test_iaf_stable_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_iaf_stable)

    def test_iaf_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_iaf)

    def test_iaf_stable_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_iaf_stable)

    def test_iaf_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_iaf)

    def test_iaf_stable_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_iaf_stable)
