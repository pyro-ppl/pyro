from functools import partial
from unittest import TestCase

import pytest
import torch

import pyro.distributions as dist
import pyro.distributions.transforms as transforms
from pyro.nn import AutoRegressiveNN, DenseNN

pytestmark = pytest.mark.init(rng_seed=123)


class FlowTests(TestCase):
    def setUp(self):
        # Epsilon is used to compare numerical gradient to analytical one
        self.epsilon = 1e-4

        # Delta is tolerance for testing f(f^{-1}(x)) = x
        self.delta = 1e-6

    def _test_jacobian(self, input_dim, make_flow):
        jacobian = torch.zeros(input_dim, input_dim)
        flow = make_flow(input_dim)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        x = torch.randn(1, input_dim)
        flow_x = flow(x)
        if flow.event_dim == 1:
            analytic_ldt = flow.log_abs_det_jacobian(x, flow_x).data
        else:
            analytic_ldt = flow.log_abs_det_jacobian(x, flow_x).sum(-1).data

        for j in range(input_dim):
            for k in range(input_dim):
                epsilon_vector = torch.zeros(1, input_dim)
                epsilon_vector[0, j] = self.epsilon
                delta = (flow(x + 0.5 * epsilon_vector) - flow(x - 0.5 * epsilon_vector)) / self.epsilon
                jacobian[j, k] = float(delta[0, k].data.sum())

        # Apply permutation for autoregressive flows with a network
        if hasattr(flow, 'arn'):
            permutation = flow.arn.get_permutation()
            permuted_jacobian = jacobian.clone()
            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]
            jacobian = permuted_jacobian

        # For autoregressive flow, Jacobian is sum of diagonal, otherwise need full determinate
        if hasattr(flow, 'autoregressive') and flow.autoregressive:
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

        # Cache is empty, hence must be calculating inverse afresh
        x_calculated = flow._inverse(y)

        assert torch.norm(x_true - x_calculated, dim=-1).max().item() < self.delta

    def _test_shape(self, base_shape, make_flow):
        base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        last_dim = base_shape[-1] if isinstance(base_shape, tuple) else base_shape
        flow = make_flow(input_dim=last_dim)
        sample = dist.TransformedDistribution(base_dist, [flow]).sample()
        assert sample.shape == base_shape

    def _make_householder(self, input_dim):
        return transforms.HouseholderFlow(input_dim, count_transforms=min(1, input_dim // 2))

    def _make_batchnorm(self, input_dim):
        # Create batchnorm transform
        bn = transforms.BatchNormTransform(input_dim)
        bn._inverse(torch.normal(torch.arange(0., input_dim), torch.arange(1., 1. + input_dim) / input_dim))
        bn.eval()
        return bn

    def _make_block_autoregressive(self, input_dim, activation='tanh', residual=None):
        return dist.BlockAutoregressive(input_dim, activation=activation, residual=residual)

    def _make_iaf(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3 * input_dim + 1])
        return transforms.InverseAutoregressiveFlow(arn)

    def _make_iaf_stable(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3 * input_dim + 1])
        return transforms.InverseAutoregressiveFlowStable(arn, sigmoid_bias=0.5)

    def _make_def(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3 * input_dim + 1], param_dims=[16]*3)
        return transforms.DeepELUFlow(arn, hidden_units=16)

    def _make_dlrf(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3 * input_dim + 1], param_dims=[16]*3)
        return transforms.DeepLeakyReLUFlow(arn, hidden_units=16)

    def _make_dsf(self, input_dim):
        arn = AutoRegressiveNN(input_dim, [3 * input_dim + 1], param_dims=[16] * 3)
        return transforms.DeepSigmoidalFlow(arn, hidden_units=16)

    def _make_permute(self, input_dim):
        permutation = torch.randperm(input_dim, device='cpu').to(torch.Tensor().device)
        return transforms.PermuteTransform(permutation)

    def _make_planar(self, input_dim):
        return transforms.PlanarFlow(input_dim)

    def _make_poly(self, input_dim):
        count_degree = 4
        count_sum = 3
        arn = AutoRegressiveNN(input_dim, [input_dim*10], param_dims=[(count_degree + 1)*count_sum])
        return transforms.PolynomialFlow(arn, input_dim=input_dim, count_degree=count_degree, count_sum=count_sum)

    def _make_affine_coupling(self, input_dim):
        split_dim = input_dim // 2
        hypernet = DenseNN(split_dim, [10*input_dim], [input_dim-split_dim, input_dim-split_dim])
        return transforms.AffineCoupling(split_dim, hypernet)

    def test_batchnorm_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_batchnorm)

    def test_block_autoregressive_jacobians(self):
        for activation in ['ELU', 'LeakyReLU', 'sigmoid', 'tanh']:
            for input_dim in [2, 3, 5, 7, 9, 11]:
                self._test_jacobian(
                    input_dim,
                    partial(
                        self._make_block_autoregressive,
                        activation=activation))

        for residual in [None, 'normal', 'gated']:
            for input_dim in [2, 3, 5, 7, 9, 11]:
                self._test_jacobian(
                    input_dim,
                    partial(
                        self._make_block_autoregressive,
                        residual=residual))

    def _make_radial(self, input_dim):
        return transforms.RadialFlow(input_dim)

    def _make_sylvester(self, input_dim):
        return transforms.SylvesterFlow(input_dim, count_transforms=input_dim//2 + 1)

    def test_iaf_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_iaf)

    def test_iaf_stable_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_iaf_stable)

    def test_def_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_def)

    def test_dlrf_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_dlrf)

    def test_dsf_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_dsf)

    def test_planar_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_planar)

    def test_poly_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_poly)

    def test_householder_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_householder)

    def test_batchnorm_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_batchnorm)

    def test_radial_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_radial)

    def test_sylvester_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_sylvester)

    def test_affine_coupling_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, self._make_affine_coupling)

    def test_iaf_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_iaf)

    def test_iaf_stable_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_iaf_stable)

    def test_permute_inverses(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_inverse(input_dim, self._make_permute)

    def test_householder_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_householder)

    def test_batchnorm_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_batchnorm)

    def test_block_autoregressive_shapes(self):
        for residual in [None, 'normal', 'gated']:
            for shape in [(3,), (3, 4), (3, 4, 2)]:
                self._test_shape(shape, partial(self._make_block_autoregressive, residual=residual))

        for activation in ['ELU', 'LeakyReLU', 'sigmoid', 'tanh']:
            for shape in [(3,), (3, 4), (3, 4, 2)]:
                self._test_shape(shape, partial(self._make_block_autoregressive, activation=activation))

    def test_iaf_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_iaf)

    def test_iaf_stable_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_iaf_stable)

    def test_def_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_def)

    def test_dlrf_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_dlrf)

    def test_dsf_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_dsf)

    def test_permute_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_permute)

    def test_planar_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_planar)

    def test_poly_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_poly)

    def test_radial_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_radial)

    def test_affine_coupling_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_affine_coupling)

    def test_sylvester_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape, self._make_sylvester)
