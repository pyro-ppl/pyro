import numpy as np
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import TestCase


class TestDelta(TestCase):
    def setUp(self):
        self.v = Variable(torch.Tensor([3]))
        self.vs = Variable(torch.Tensor([[0], [1], [2], [3]]))
        self.test_data = Variable(torch.Tensor([3, 3, 3]))
        self.batch_test_data = Variable(torch.arange(0, 4).unsqueeze(1).expand(4, 3))
        self.analytic_mean = 3
        self.analytic_var = 0
        self.n_samples = 10

    def test_log_pdf(self):
        log_px_torch = dist.delta.log_pdf(self.test_data, self.v).data
        self.assertEqual(torch.sum(log_px_torch), 0)

    def test_batch_log_pdf(self):
        log_px_torch = dist.delta.batch_log_pdf(self.batch_test_data, self.vs, batch_size=2).data
        self.assertEqual(torch.sum(log_px_torch), 0)

    def test_mean_and_var(self):
        torch_samples = [dist.delta(self.v).data.cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(torch_mean, self.analytic_mean)
        self.assertEqual(torch_var, self.analytic_var)
