from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np
import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.iaf import InverseAutoregressiveFlow
from pyro.nn import AutoRegressiveNN
from pyro.nn.auto_reg_nn import create_mask

pytestmark = pytest.mark.init(rng_seed=123)


class InverseAutoregressiveFlowTests(TestCase):
    def setUp(self):
        self.epsilon = 1.0e-3

    def _test_jacobian(self, input_dim, hidden_dim):
        jacobian = torch.zeros(input_dim, input_dim)
        iaf = InverseAutoregressiveFlow(AutoRegressiveNN(input_dim, [40]), sigmoid_bias=0.5)

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

    def _test_shape(self, base_shape):
        base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        last_dim = base_shape[-1] if isinstance(base_shape, tuple) else base_shape
        iaf = InverseAutoregressiveFlow(AutoRegressiveNN(last_dim, [40]))
        sample = dist.TransformedDistribution(base_dist, [iaf]).sample()
        assert sample.shape == base_shape

    def test_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, 3 * input_dim + 1)

    def test_shapes(self):
        for shape in [(3,), (3, 4), (3, 4, 2)]:
            self._test_shape(shape)


class AutoRegressiveNNTests(TestCase):
    def setUp(self):
        self.epsilon = 1.0e-3

    def _test_jacobian(self, input_dim, hidden_dim, param_dim):
        jacobian = torch.zeros(input_dim, input_dim)
        arn = AutoRegressiveNN(input_dim, [hidden_dim], param_dims=[param_dim])

        def nonzero(x):
            return torch.sign(torch.abs(x))

        for output_index in range(param_dim):
            for j in range(input_dim):
                for k in range(input_dim):
                    x = torch.randn(1, input_dim)
                    epsilon_vector = torch.zeros(1, input_dim)
                    epsilon_vector[0, j] = self.epsilon
                    delta = (arn(x + 0.5 * epsilon_vector) - arn(x - 0.5 * epsilon_vector)) / self.epsilon
                    jacobian[j, k] = float(delta[0, output_index, k])

            permutation = arn.get_permutation()
            permuted_jacobian = jacobian.clone()
            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]

            lower_sum = torch.sum(torch.tril(nonzero(permuted_jacobian), diagonal=0))

            assert lower_sum == float(0.0)

    def _test_masks(self, input_dim, observed_dim, hidden_dims, permutation, output_dim_multiplier):
        masks, mask_skip = create_mask(input_dim, observed_dim, hidden_dims, permutation, output_dim_multiplier)

        # First test that hidden layer masks are adequately connected
        # Tracing backwards, works out what inputs each output is connected to
        # It's a dictionary of sets indexed by a tuple (input_dim, param_dim)
        permutation = list(permutation.numpy())

        # Loop over variables
        for idx in range(input_dim):
            # Calculate correct answer
            correct = torch.cat((torch.arange(observed_dim, dtype=torch.long), torch.tensor(
                sorted(permutation[0:permutation.index(idx)]), dtype=torch.long) + observed_dim))

            # Loop over parameters for each variable
            for jdx in range(output_dim_multiplier):
                prev_connections = set()
                # Do output-to-penultimate hidden layer mask
                for kdx in range(masks[-1].size(1)):
                    if masks[-1][idx + jdx * input_dim, kdx]:
                        prev_connections.add(kdx)

                # Do hidden-to-hidden, and hidden-to-input layer masks
                for m in reversed(masks[:-1]):
                    this_connections = set()
                    for kdx in prev_connections:
                        for ldx in range(m.size(1)):
                            if m[kdx, ldx]:
                                this_connections.add(ldx)
                    prev_connections = this_connections

                assert (torch.tensor(list(sorted(prev_connections)), dtype=torch.long) == correct).all()

                # Test the skip-connections mask
                skip_connections = set()
                for kdx in range(mask_skip.size(1)):
                    if mask_skip[idx + jdx * input_dim, kdx]:
                        skip_connections.add(kdx)
                assert (torch.tensor(list(sorted(skip_connections)), dtype=torch.long) == correct).all()

    def test_jacobians(self):
        for input_dim in [2, 3, 5, 7, 9, 11]:
            self._test_jacobian(input_dim, 3 * input_dim + 1, 2)

    def test_masks(self):
        for input_dim in [1, 3, 5]:
            for observed_dim in [0, 3]:
                for num_layers in [1, 3]:
                    for output_dim_multiplier in [1, 2, 3]:
                        # NOTE: the hidden dimension must be greater than the input_dim for the
                        # masks to be well-defined!
                        hidden_dim = input_dim * 5
                        permutation = torch.randperm(input_dim, device='cpu')
                        self._test_masks(
                            input_dim,
                            observed_dim,
                            [hidden_dim]*num_layers,
                            permutation,
                            output_dim_multiplier)
