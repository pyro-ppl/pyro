from __future__ import absolute_import, division, print_function

from unittest import TestCase

import pytest
import torch

import pyro.distributions as dist
from pyro.nn import AutoRegressiveNN

pytestmark = pytest.mark.init(rng_seed=123)


class FlowTests(TestCase):
    def setUp(self):
        # Epsilon is used to compare numerical gradient to analytical one
        self.epsilon = 1e-3

        # Delta is tolerance for testing f(f^{-1}(x)) = x
        self.delta = 1e-6

    def _test_jacobian(self, input_dim, make_flow):
        jacobian = torch.zeros(input_dim, input_dim)
        flow = make_flow(input_dim)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        x = torch.randn(1, input_dim)
        flow_x = flow(x)
        analytic_ldt = flow.log_abs_det_jacobian(x, flow_x).data.sum()

        for j in range(input_dim):
            for k in range(input_dim):
                epsilon_vector = torch.zeros(1, input_dim)
                epsilon_vector[0, j] = self.epsilon
                delta = (flow(x + 0.5 * epsilon_vector) - flow(x - 0.5 * epsilon_vector)) / self.epsilon
                jacobian[j, k] = float(delta[0, k].data.sum())

        # Apply permutation for autoregressive flows
        if hasattr(flow, 'arn'):
            permutation = flow.arn.get_permutation()
            permuted_jacobian = jacobian.clone()
            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]
            jacobian = permuted_jacobian

        # For autoregressive flow, Jacobian is sum of diagonal, otherwise need full determinate
        if hasattr(flow, 'arn'):
            numeric_ldt = torch.sum(torch.log(torch.diag(jacobian)))
        else:
            numeric_ldt = torch.log(torch.abs(jacobian.det()))

        ldt_discrepancy = (analytic_ldt - numeric_ldt).abs()
        assert ldt_discrepancy < self.epsilon

        # Test that lower triangular with unit diagonal for autoregressive flows
        if hasattr(flow, 'arn'):
            diag_sum = torch.sum(torch.diag(nonzero(jacobian)))
            lower_sum = torch.sum(torch.tril(nonzero(jacobian), diagonal=-1))
            assert diag_sum == float(input_dim)
            assert lower_sum == float(0.0)

    def _test_inverse(self, input_dim, make_flow):
        base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
        flow = make_flow(input_dim)

        x_true = base_dist.sample(torch.Size([10]))
        y = flow._call(x_true)

        # This line empties the inverse cache, if the flow uses it
        if hasattr(flow, '_intermediates_cache'):
            flow._intermediates_cache.pop((y, 'log_scale'))
            flow._intermediates_cache.pop((y, 'x'))

        # Cache is empty, hence must be calculating inverse afresh
        x_calculated = flow._inverse(y)

        assert torch.norm(x_true - x_calculated, dim=-1).max().item() < self.delta

    def _test_shape(self, base_shape, make_flow):
        base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        last_dim = base_shape[-1] if isinstance(base_shape, tuple) else base_shape
        flow = make_flow(input_dim=last_dim)
        sample = dist.TransformedDistribution(base_dist, [flow]).sample()
        assert sample.shape == base_shape

    def _make_iaf(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3 * input_dim + 1])
        return dist.InverseAutoregressiveFlow(arn)

    def _make_iaf_stable(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3 * input_dim + 1])
        return dist.InverseAutoregressiveFlowStable(arn, sigmoid_bias=0.5)

    def _make_flipflow(self, input_dim):
        permutation = torch.randperm(input_dim, device='cpu').to(torch.Tensor().device)
        return dist.PermuteTransform(permutation)

    def _make_planar(self, input_dim):
        return dist.PlanarFlow(input_dim)

    def test_iaf_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_iaf)

    def test_iaf_stable_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_iaf_stable)

    def test_planar_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_planar)

    def test_iaf_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_iaf)

    def test_iaf_stable_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_iaf_stable)

    def test_flipflow_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_flipflow)

    def test_iaf_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_iaf)

    def test_iaf_stable_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_iaf_stable)

    def test_flipflow_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_flipflow)

    def test_planar_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_planar)
