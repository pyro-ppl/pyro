from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import TestCase, assert_equal


class TestDelta(TestCase):
    def setUp(self):
        self.v = Variable(torch.Tensor([3]))
        self.vs = Variable(torch.Tensor([[0], [1], [2], [3]]))
        self.vs_expanded = self.vs.expand(4, 3)
        self.test_data = Variable(torch.Tensor([[3], [3], [3]]))
        self.batch_test_data_1 = Variable(torch.arange(0, 4).unsqueeze(1).expand(4, 3))
        self.batch_test_data_2 = Variable(torch.arange(4, 8).unsqueeze(1).expand(4, 3))
        self.batch_test_data_3 = Variable(torch.Tensor([[3], [3], [3], [3]]))
        self.expected_support = [[0], [1], [2], [3]]
        self.expected_support_non_vec = [3]
        self.analytic_mean = 3
        self.analytic_var = 0
        self.n_samples = 10

    def test_log_pdf(self):
        log_px_torch = dist.delta.log_pdf(self.test_data, self.v).data
        self.assertEqual(torch.sum(log_px_torch), 0)

    def test_batch_log_pdf(self):
        log_px_torch = dist.delta.batch_log_pdf(self.batch_test_data_1, self.vs_expanded).data
        self.assertEqual(torch.sum(log_px_torch), 0)
        log_px_torch = dist.delta.batch_log_pdf(self.batch_test_data_2, self.vs_expanded).data
        self.assertEqual(torch.sum(log_px_torch), float('-inf'))

    def test_batch_log_pdf_shape(self):
        assert dist.delta.batch_log_pdf(self.batch_test_data_3, self.vs).size() == (4, 1)
        assert dist.delta.batch_log_pdf(self.batch_test_data_3, self.v).size() == (4, 1)

    def test_mean_and_var(self):
        torch_samples = [dist.delta(self.v).data.cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(torch_mean, self.analytic_mean)
        self.assertEqual(torch_var, self.analytic_var)

    def test_support(self):
        actual_support = dist.delta.enumerate_support(self.vs)
        actual_support_non_vec = dist.delta.enumerate_support(self.v)
        assert_equal(actual_support.data, torch.Tensor(self.expected_support))
        assert_equal(actual_support_non_vec.data, torch.Tensor(self.expected_support_non_vec))
