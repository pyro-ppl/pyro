from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np
import pytest
import torch

from pyro.distributions.iaf import InverseAutoregressiveFlow
from pyro.nn import AutoRegressiveNN

pytestmark = pytest.mark.init(rng_seed=123)


class InverseAutoregressiveFlowTests(TestCase):
    def setUp(self):
        self.epsilon = 1.0e-6

    def _test_jacobian(self, input_dim, hidden_dim):
        jacobian = torch.zeros(input_dim, input_dim)
        iaf = InverseAutoregressiveFlow(input_dim, hidden_dim, sigmoid_bias=0.5)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        x = torch.randn(1, input_dim)
        iaf_x = iaf(x)
        analytic_ldt = iaf.log_abs_det_jacobian(x, iaf_x).data.sum()

        for j in range(input_dim):
            for k in range(input_dim):
                epsilon_vector = torch.zeros(1, input_dim)
                epsilon_vector[0, j] = self.epsilon
                iaf_x_eps = iaf(x + epsilon_vector)
                delta = (iaf_x_eps - iaf_x) / self.epsilon
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

    def test_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, 3 * input_dim + 1)


class AutoRegressiveNNTests(TestCase):

    def setUp(self):
        self.epsilon = 1.0e-6

    def _test_jacobian(self, input_dim, hidden_dim, multiplier):
        jacobian = torch.zeros(input_dim, input_dim)
        arn = AutoRegressiveNN(input_dim, hidden_dim, multiplier)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        for output_index in range(multiplier):
            for j in range(input_dim):
                for k in range(input_dim):
                    x = torch.randn(1, input_dim)
                    epsilon_vector = torch.zeros(1, input_dim)
                    epsilon_vector[0, j] = self.epsilon
                    delta = (arn(x + epsilon_vector) - arn(x)) / self.epsilon
                    jacobian[j, k] = float(delta[0, k + output_index * input_dim])

            permutation = arn.get_permutation()
            permuted_jacobian = jacobian.clone()
            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]

            lower_sum = torch.sum(torch.tril(nonzero(permuted_jacobian), diagonal=0))
            assert lower_sum == float(0.0)

    def test_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, 3 * input_dim + 1, 2)
