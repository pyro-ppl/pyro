import json

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import TestCase

pytestmark = pytest.mark.init(rng_seed=123)


class TestCategorical(TestCase):
    """
    Tests methods specific to the Categorical distribution like nhot_encoding
    """

    def setUp(self):
        n = 1
        self.ps = Variable(torch.Tensor([0.1, 0.6, 0.3]))
        self.batch_ps = Variable(torch.Tensor([[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]]))
        self.n = Variable(torch.Tensor([n]))
        self.test_data = Variable(torch.Tensor([0, 1, 0]))
        self.test_data_nhot = Variable(torch.Tensor([2]))
        self.analytic_mean = n * self.ps
        one = Variable(torch.ones(3))
        self.analytic_var = n * torch.mul(self.ps, one.sub(self.ps))

        # Discrete Distribution
        self.d_ps = Variable(torch.Tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]))
        self.d_vs = Variable(torch.Tensor([[0, 1, 2], [3, 4, 5]]))
        self.d_vs_arr = [['a', 'b', 'c'], ['d', 'e', 'f']]
        self.d_vs_tup = (('a', 'b', 'c'), ('d', 'e', 'f'))
        self.d_test_data = Variable(torch.Tensor([[0], [5]]))
        self.d_v_test_data = [['a'], ['f']]

        self.n_samples = 50000

        with open('tests/test_data/support_categorical.json') as data_file:
            data = json.load(data_file)
        self.support = [torch.Tensor(x) for x in data['one_hot']]
        self.nhot_support = list(map(lambda x: torch.Tensor(x), data['not_hot']))
        self.discrete_support = list(map(lambda x: torch.Tensor(x), data['discrete']))
        self.discrete_arr_support = data['discrete_arr']

    def test_nhot_log_pdf(self):
        log_px_torch = dist.categorical.batch_log_pdf(self.test_data_nhot,
                                                      self.ps,
                                                      one_hot=False,
                                                      batch_size=1).data[0][0]
        log_px_np = float(sp.multinomial.logpmf(np.array([0, 0, 1]), 1, self.ps.data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [dist.categorical(self.ps, one_hot=False, batch_size=1).data.numpy()
                         for _ in range(self.n_samples)]
        _, counts = np.unique(torch_samples, return_counts=True)
        computed_mean = float(counts[0]) / self.n_samples
        self.assertEqual(computed_mean, self.analytic_mean.data.numpy()[0], prec=0.05)

    def test_discrete_log_pdf(self):
        log_px_torch = dist.categorical.batch_log_pdf(self.d_test_data, self.d_ps, self.d_vs).data[0][0]
        log_px_np = float(sp.multinomial.logpmf(np.array([1, 0, 0]), 1, self.d_ps[0].data.numpy()))
        log_px_torch2 = dist.categorical.batch_log_pdf(self.d_test_data, self.d_ps, self.d_vs).data[1][0]
        log_px_np2 = float(sp.multinomial.logpmf(np.array([0, 0, 1]), 1, self.d_ps[1].data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)
        self.assertEqual(log_px_torch2, log_px_np2, prec=1e-4)

    def test_discrete_arr_logpdf(self):
        log_px_torch = dist.categorical.batch_log_pdf(self.d_v_test_data,
                                                      self.d_ps, self.d_vs_arr).data[0][0]
        log_px_np = float(sp.multinomial.logpmf(np.array([1, 0, 0]), 1, self.d_ps[0].data.numpy()))
        log_px_torch2 = dist.categorical.batch_log_pdf(self.d_v_test_data,
                                                       self.d_ps, self.d_vs_arr).data[1][0]
        log_px_np2 = float(sp.multinomial.logpmf(np.array([0, 0, 1]), 1, self.d_ps[1].data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)
        self.assertEqual(log_px_torch2, log_px_np2, prec=1e-4)

    def test_discrete_support(self):
        s = list(dist.categorical.support(self.d_ps, self.d_vs))
        v = [torch.equal(x.data, y) for x, y in zip(s, self.discrete_support)]
        assert all(v)

    def test_discrete_arr_support(self):
        s = list(dist.categorical.support(self.d_ps, self.d_vs_arr))
        assert s == self.discrete_arr_support

    def test_nhot_support(self):
        s = list(dist.categorical.support(self.batch_ps, one_hot=False))
        v = [torch.equal(x.data, y) for x, y in zip(s, self.nhot_support)]
        assert all(v)
