from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np
import torch

import pyro.distributions as dist
from tests.common import assert_equal


class TestDelta(TestCase):
    def setUp(self):
        self.v = torch.tensor([3])
        self.vs = torch.tensor([[0], [1], [2], [3]])
        self.vs_expanded = self.vs.expand(4, 3)
        self.test_data = torch.tensor([[3], [3], [3]])
        self.batch_test_data_1 = torch.arange(0, 4).unsqueeze(1).expand(4, 3)
        self.batch_test_data_2 = torch.arange(4, 8).unsqueeze(1).expand(4, 3)
        self.batch_test_data_3 = torch.Tensor([[3], [3], [3], [3]])
        self.expected_support = [[[0], [1], [2], [3]]]
        self.expected_support_non_vec = [[3]]
        self.analytic_mean = 3
        self.analytic_var = 0
        self.n_samples = 10

    def test_log_pdf(self):
        log_px_torch = dist.Delta(self.v).log_prob(self.test_data).sum()
        assert_equal(log_px_torch.item(), 0)

    def test_batch_log_prob(self):
        log_px_torch = dist.Delta(self.vs_expanded).log_prob(self.batch_test_data_1).data
        assert_equal(log_px_torch.sum().item(), 0)
        log_px_torch = dist.Delta(self.vs_expanded).log_prob(self.batch_test_data_2).data
        assert_equal(log_px_torch.sum().item(), float('-inf'))

    def test_batch_log_prob_shape(self):
        assert dist.Delta(self.vs).log_prob(self.batch_test_data_3).size() == (4, 1)
        assert dist.Delta(self.v).log_prob(self.batch_test_data_3).size() == (4, 1)

    def test_mean_and_var(self):
        torch_samples = [dist.Delta(self.v).sample().detach().cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        assert_equal(torch_mean, self.analytic_mean)
        assert_equal(torch_var, self.analytic_var)

    def test_support(self):
        actual_support = dist.Delta(self.vs).enumerate_support()
        actual_support_non_vec = dist.Delta(self.v).enumerate_support()
        assert len(actual_support) == 1
        assert len(actual_support_non_vec) == 1
        assert_equal(actual_support.data, torch.tensor(self.expected_support))
        assert_equal(actual_support_non_vec.data, torch.tensor(self.expected_support_non_vec))
