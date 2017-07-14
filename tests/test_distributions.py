import numpy as np
import scipy.stats as spr
import json
import itertools

import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import TestCase
from pyro.distributions.transformed_distribution import AffineExp, TransformedDistribution


class TestUniform(TestCase):

    def setUp(self):
        self.a = Variable(torch.Tensor([-1.0]))
        self.b = Variable(torch.Tensor([2.0]))
        self.vec_a = Variable(torch.Tensor([[2.0], [-3], [0]]))
        self.vec_b = Variable(torch.Tensor([[2.5], [0], [1]]))

        self.a_np = -1
        self.b_np = 2
        self.vec_a_np = self.vec_a.data.cpu().numpy()
        self.vec_b_np = self.vec_b.data.cpu().numpy()
        self.range_np = self.b_np - self.a_np
        self.vec_range_np = self.vec_b_np - self.vec_a_np
        self.test_data = Variable(torch.Tensor([-0.5]))
        self.batch_test_data = Variable(torch.Tensor([[1], [-2], [0.7]]))

        self.g = dist.Uniform(self.a, self.b)
        self.batch_g = dist.Uniform(self.vec_a, self.vec_b)

    def test_log_pdf(self):
        log_px_torch = self.g.log_pdf(self.test_data).data[0]
        log_px_np = spr.uniform.logpdf(self.test_data.data.cpu().numpy(),
                                       loc=self.a_np,
                                       scale=self.range_np)[0]
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_g.batch_log_pdf(self.batch_test_data).data
        log_px_np = spr.uniform.logpdf(self.batch_test_data.data.cpu().numpy(),
                                       loc=self.vec_a_np,
                                       scale=self.vec_range_np)
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)


class TestExponential(TestCase):

    def setUp(self):
        self.lam = Variable(torch.Tensor([2.4]))
        self.batch_lam = Variable(torch.Tensor([[2.4], [1.4]]))
        self.test_data = Variable(torch.Tensor([5.5]))
        self.test_batch_data = Variable(torch.Tensor([[5.5], [3.2]]))
        self.dist = dist.Exponential(self.lam)
        self.batch_dist = dist.Exponential(self.batch_lam)
        self.analytic_mean = torch.pow(self.lam, -1.0).data.cpu().numpy()[0]
        self.analytic_var = torch.pow(self.lam, -2.0).data.cpu().numpy()[0]
        self.n_samples = 10000

    def test_log_pdf(self):
        log_px_torch = self.dist.log_pdf(self.test_data).data[0]
        log_px_np = spr.expon.logpdf(
            self.test_data.data.cpu().numpy(),
            scale=1.0 / self.lam.data.cpu().numpy())
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_dist.batch_log_pdf(self.test_batch_data).data[0]
        log_px_np = spr.expon.logpdf(
            self.test_data.data.cpu().numpy(),
            scale=1.0 / self.lam.data.cpu().numpy())
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [self.dist.sample().data.cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(torch_mean, self.analytic_mean, prec=0.05)
        self.assertEqual(torch_var, self.analytic_var, prec=0.05)


class TestGamma(TestCase):

    def setUp(self):
        self.alpha = Variable(torch.Tensor([2.4]))
        self.batch_alpha = Variable(torch.Tensor([[2.4], [3.2]]))
        self.batch_beta = Variable(torch.Tensor([[np.sqrt(2.4)], [np.sqrt(3.2)]]))
        self.beta = Variable(torch.Tensor([np.sqrt(2.4)]))
        self.test_data = Variable(torch.Tensor([5.5]))
        self.batch_test_data = Variable(torch.Tensor([[5.5], [4.4]]))
        self.dist = dist.Gamma(self.alpha, self.beta)
        self.batch_dist = dist.Gamma(self.batch_alpha, self.batch_beta)
        self.analytic_mean = (self.alpha / self.beta).data.cpu().numpy()[0]
        self.analytic_var = (
            self.alpha /
            torch.pow(
                self.beta,
                2.0)).data.cpu().numpy()[0]
        self.n_samples = 50000

    def test_log_pdf(self):
        log_px_torch = self.dist.log_pdf(self.test_data).data[0]
        log_px_np = spr.gamma.logpdf(
            self.test_data.data.cpu().numpy(),
            self.alpha.data.cpu().numpy(),
            scale=1.0 / self.beta.data.cpu().numpy())
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_dist.batch_log_pdf(self.batch_test_data).data[0]
        log_px_np = spr.gamma.logpdf(
            self.test_data.data.cpu().numpy(),
            self.alpha.data.cpu().numpy(),
            scale=1.0 / self.beta.data.cpu().numpy())
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [self.dist.sample().data.cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(torch_mean, self.analytic_mean, prec=0.05)
        self.assertEqual(torch_var, self.analytic_var, prec=0.05)


class TestMultinomial(TestCase):
    def setUp(self):
        n = 8
        self.ps = Variable(torch.Tensor([0.1, 0.6, 0.3]))
        self.n = Variable(torch.Tensor([n]))
        self.batch_ps = Variable(torch.Tensor([[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]]))
        self.batch_n = Variable(torch.Tensor([[n]]))
#         self.test_data = Variable(torch.Tensor([0, 0, 1, 1, 2, 1, 1, 2]))
        self.test_data = Variable(torch.Tensor([2, 4, 2]))
        self.batch_test_data = Variable(torch.Tensor([[2, 4, 2], [1, 4, 3]]))
        self.dist = dist.Multinomial(self.ps, self.n)
        self.batch_dist = dist.Multinomial(self.batch_ps, self.batch_n, batch_size=1)
        self.analytic_mean = n * self.ps
        one = Variable(torch.ones(3))
        self.analytic_var = n * torch.mul(self.ps, one.sub(self.ps))
        self.n_samples = 50000

    def test_log_pdf(self):
        log_px_torch = self.dist.log_pdf(self.test_data).data[0]
        log_px_np = float(spr.multinomial.logpmf(np.array([2, 4, 2]), 8, self.ps.data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_dist.batch_log_pdf(self.batch_test_data).data.numpy()
        log_px_np0 = float(spr.multinomial.logpmf(np.array([2, 4, 2]), 8, self.ps.data.numpy()))
        log_px_np1 = float(spr.multinomial.logpmf(np.array([1, 4, 3]), 8, np.array([0.2, 0.4, 0.4])))
        log_px_np = [log_px_np0, log_px_np1]
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [self.dist.expanded_sample().data.numpy() for _ in range(self.n_samples)]
        _, counts = np.unique(torch_samples, return_counts=True)
        exp_ = float(counts[0]) / self.n_samples
        torch_var = float(counts[0]) * np.power(0.1 * (0 - np.mean(torch_samples)), 2)
        torch_var = np.square(np.mean(torch_samples)) / 2
        self.assertEqual(exp_, self.analytic_mean.data.numpy()[0], prec=0.05)
        self.assertEqual(torch_var, self.analytic_var.data.numpy()[0], prec=0.05)


class TestCategorical(TestCase):
    def setUp(self):
        n = 1
        self.ps = Variable(torch.Tensor([0.1, 0.6, 0.3]))
        self.batch_ps = Variable(torch.Tensor([[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]]))
        self.n = Variable(torch.Tensor([n]))
#         self.test_data = Variable(torch.Tensor([0, 0, 1, 1, 2, 1, 1, 2]))
        self.test_data = Variable(torch.Tensor([0, 1, 0]))
        self.test_data_nhot = Variable(torch.Tensor([2]))
        self.dist = dist.Categorical(self.ps, batch_size=1)
        self.dist_nhot = dist.Categorical(self.ps, one_hot=False, batch_size=1)
        self.batch_dist = dist.Categorical(self.batch_ps, batch_size=1)
        self.batch_dist_nhot = dist.Categorical(self.batch_ps, one_hot=False, batch_size=1)
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
        self.d_dist = dist.Categorical(self.d_ps, self.d_vs)
        self.d_dist_arr = dist.Categorical(self.d_ps, self.d_vs_arr)
        self.d_dist_tup = dist.Categorical(self.d_ps, self.d_vs_tup)

        self.n_samples = 50000

        with open('tests/test_data/support_categorical.json') as data_file:
            data = json.load(data_file)
        self.support = list(map(lambda x: torch.Tensor(x), data['one_hot']))
        self.nhot_support = list(map(lambda x: torch.Tensor(x), data['not_hot']))
        self.discrete_support = list(map(lambda x: torch.Tensor(x), data['discrete']))
        self.discrete_arr_support = data['discrete_arr']

    def test_nhot_log_pdf(self):
        log_px_torch = self.dist_nhot.batch_log_pdf(self.test_data_nhot).data[0][0]
        log_px_np = float(spr.multinomial.logpmf(np.array([0, 0, 1]), 1, self.ps.data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_log_pdf(self):
        log_px_torch = self.dist.log_pdf(self.test_data).data[0]
        log_px_np = float(spr.multinomial.logpmf(np.array([0, 1, 0]), 1, self.ps.data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.dist.batch_log_pdf(self.test_data).data[0][0]
        log_px_np = float(spr.multinomial.logpmf(np.array([0, 1, 0]), 1, self.ps.data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [self.dist_nhot.sample().data.numpy() for _ in range(self.n_samples)]
        _, counts = np.unique(torch_samples, return_counts=True)
        exp_ = float(counts[0]) / self.n_samples
        torch_var = float(counts[0]) * np.power(0.1 * (0 - np.mean(torch_samples)), 2)
        torch_var = np.square(np.mean(torch_samples)) / 16
        self.assertEqual(exp_, self.analytic_mean.data.numpy()[0], prec=0.05)
        self.assertEqual(torch_var, self.analytic_var.data.numpy()[0], prec=0.05)

    def test_discrete_log_pdf(self):
        log_px_torch = self.d_dist.batch_log_pdf(self.d_test_data).data[0][0]
        log_px_np = float(spr.multinomial.logpmf(np.array([1, 0, 0]), 1, self.d_ps[0].data.numpy()))
        log_px_torch2 = self.d_dist.batch_log_pdf(self.d_test_data).data[1][0]
        log_px_np2 = float(spr.multinomial.logpmf(np.array([0, 0, 1]), 1, self.d_ps[1].data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)
        self.assertEqual(log_px_torch2, log_px_np2, prec=1e-4)

    def test_discrete_arr_logpdf(self):
        log_px_torch = self.d_dist_arr.batch_log_pdf(self.d_v_test_data).data[0][0]
        log_px_np = float(spr.multinomial.logpmf(np.array([1, 0, 0]), 1, self.d_ps[0].data.numpy()))
        log_px_torch2 = self.d_dist.batch_log_pdf(self.d_test_data).data[1][0]
        log_px_np2 = float(spr.multinomial.logpmf(np.array([0, 0, 1]), 1, self.d_ps[1].data.numpy()))
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)
        self.assertEqual(log_px_torch2, log_px_np2, prec=1e-4)

    def test_discrete_support(self):
        s = list(self.d_dist.support())
        v = [torch.equal(x.data, y) for x, y in zip(s, self.discrete_support)]
        self.assertTrue(all(v))

    def test_discrete_arr_support(self):
        s = list(self.d_dist_arr.support())
        self.assertTrue(s == self.discrete_arr_support)

    def test_support(self):
        s = list(self.batch_dist.support())
        v = [torch.equal(x.data, y) for x, y in zip(s, self.support)]
        self.assertTrue(all(v))

    def test_nhot_support(self):
        s = list(self.batch_dist_nhot.support())
        v = [torch.equal(x.data, y) for x, y in zip(s, self.nhot_support)]
        self.assertTrue(all(v))


class TestBeta(TestCase):
    def setUp(self):
        self.alpha = Variable(torch.Tensor([2.4]))
        self.beta = Variable(torch.Tensor([3.7]))
        self.test_data = Variable(torch.Tensor([0.4]))
        self.dist = dist.Beta(self.alpha, self.beta)
        self.batch_alpha = Variable(torch.Tensor([[2.4], [3.6]]))
        self.batch_beta = Variable(torch.Tensor([[3.7], [2.5]]))
        self.batch_test_data = Variable(torch.Tensor([[0.4], [0.6]]))
        self.batch_dist = dist.Beta(self.batch_alpha, self.batch_beta)
        self.analytic_mean = (self.alpha / (self.alpha + self.beta))
        one = Variable(torch.ones([1]))
        self.analytic_var = torch.pow(
            self.analytic_mean, 2.0) * self.beta / (self.alpha * (self.alpha + self.beta + one))
        self.n_samples = 50000

    def test_log_pdf(self):
        log_px_torch = self.dist.log_pdf(self.test_data).data[0]
        log_px_np = spr.beta.logpdf(
            self.test_data.data.cpu().numpy(),
            self.alpha.data.cpu().numpy(),
            self.beta.data.cpu().numpy())
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_dist.batch_log_pdf(self.batch_test_data).data[0]
        log_px_np = spr.beta.logpdf(
            self.test_data.data.cpu().numpy(),
            self.alpha.data.cpu().numpy(),
            self.beta.data.cpu().numpy())
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [self.dist.sample().data.cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(
            torch_mean,
            self.analytic_mean.data.cpu().numpy()[0],
            prec=0.05)
        self.assertEqual(
            torch_var,
            self.analytic_var.data.cpu().numpy()[0],
            prec=0.05)


class TestPoisson(TestCase):

    def setUp(self):
        self.lam = Variable(torch.Tensor([2, 4.5, 3., 5.1]))
        self.dim = 4
        self.batch_lam = Variable(torch.Tensor([[2, 4.5, 3., 5.1], [6, 3.2, 1, 4]]))
        self.test_data = Variable(torch.Tensor([0, 1, 2, 4]))
        self.batch_test_data = Variable(torch.Tensor([[0, 1, 2, 4], [4, 1, 2, 4]]))
        self.g = dist.Poisson(self.lam)
        self.batch_g = dist.Poisson(self.batch_lam)

    def test_log_pdf(self):
        log_px_torch = self.g.log_pdf(self.test_data).data[0]
        log_px_np_ = [
            spr.poisson.logpmf(
                self.test_data.data.cpu().numpy()[i],
                self.lam.data.cpu().numpy()[i]) for i in range(
                self.dim)]
        log_px_np = np.sum(log_px_np_)
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_g.batch_log_pdf(self.batch_test_data).data[0]
        log_px_np_ = [
            spr.poisson.logpmf(
                self.test_data.data.cpu().numpy()[i],
                self.lam.data.cpu().numpy()[i]) for i in range(
                self.dim)]
        log_px_np = np.sum(log_px_np_)
        self.assertEqual(log_px_torch[0], log_px_np, prec=1e-4)


class TestNormalChol(TestCase):

    def setUp(self):
        self.mu = Variable(torch.ones(2))
        self.L = Variable(torch.Tensor([[2.0, 0.0], [1.0, 3.0]]))

        self.mu_np = self.mu.data.cpu().numpy()
        self.L_np = self.L.data.cpu().numpy()
        self.cov_np = np.matmul(self.L_np, np.transpose(self.L_np))
        self.test_data = Variable(torch.randn(2))
        self.g = dist.Normal_Chol(self.mu, self.L)

    def test_log_pdf(self):
        log_px_torch = self.g.log_pdf(self.test_data).data[0]
        log_px_np = spr.multivariate_normal.logpdf(self.test_data.data.cpu().numpy(),
                                                   mean=self.mu_np,
                                                   cov=self.cov_np)
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)


class TestNormal(TestCase):

    def setUp(self):
        self.mu = Variable(torch.ones(2))
        self.L = Variable(torch.Tensor([[2.0, 0.0], [1.0, 3.0]]))

        self.mu_np = self.mu.data.cpu().numpy()
        self.L_np = self.L.data.cpu().numpy()
        self.cov_np = np.matmul(self.L_np, np.transpose(self.L_np))
        self.cov_torch = Variable(torch.from_numpy(self.cov_np))

        self.test_data = Variable(torch.randn(2))

        self.g = dist.Normal(self.mu, self.cov_torch)

    def test_log_pdf(self):
        log_px_torch = self.g.log_pdf(self.test_data).data[0]
        log_px_np = spr.multivariate_normal.logpdf(self.test_data.data.cpu().numpy(),
                                                   mean=self.mu_np,
                                                   cov=self.cov_np)
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)


class TestDiagNormal(TestCase):

    def setUp(self):
        self.mu = Variable(torch.ones(3))
        self.sigma = 2 * Variable(torch.ones(3))
        self.mu_np = self.mu.data.cpu().numpy()
        self.sigma_np = self.sigma.data.cpu().numpy()
        self.test_data = Variable(torch.randn(3))
        self.g = dist.DiagNormal(self.mu, self.sigma)

        self.batch_mu = Variable(torch.ones(2, 50))
        self.batch_sigma = 2 * Variable(torch.ones(2, 50))
        self.batch_mu_np = self.mu.data.cpu().numpy()
        self.batch_sigma_np = self.sigma.data.cpu().numpy()
        self.batch_test_data = Variable(torch.randn(2, 50))
        self.batch_g = dist.DiagNormal(self.mu, self.sigma, batch_size=1)


    def test_log_pdf(self):
        log_px_torch = self.g.log_pdf(self.test_data).data[0]
        log_px_np = spr.multivariate_normal.logpdf(self.test_data.data.cpu().numpy(),
                                                   mean=self.mu_np,
                                                   cov=self.sigma_np ** 2.0)
        self.assertEqual(log_px_torch, log_px_np, prec=1e-3)

    def test_batch_log_pdf(self):
        bs = 10
        log_px_torch = self.g.batch_log_pdf(self.batch_test_data, bs).data[0]
        log_px_np = spr.multivariate_normal.logpdf(self.tbatch_est_data.data.cpu().numpy()[0],
                                                   mean=self.batch_mu_np[0],
                                                   cov=self.batch_sigma_np[0] ** 2.0)
        self.assertEqual(log_px_torch[0], log_px_np, prec=2e-2)


class TestBernoulli(TestCase):

    def setUp(self):
        self.ps = Variable(torch.Tensor(
            [0.25, 0.5, 0.75, 0.5, 0.25, 0.3, 0.1, 0.8, 0.9, 0.6]))
        self.small_ps = Variable(torch.Tensor(
            [[0.25, 0.5, 0.75], [0.3, 0.6, 0.1]]))
        self.ps_np = self.ps.data.cpu().numpy()
        self.test_data = Variable(torch.Tensor([[1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                                                [1, 0, 0, 1, 0, 0, 1, 1, 0, 1]]))
        self.g_1 = dist.Bernoulli(self.ps)
        self.g = dist.Bernoulli(self.ps, batch_size=2)
        self.small_g = dist.Bernoulli(self.small_ps)

        with open('tests/test_data/support_bernoulli.json') as data_file:
            data = json.load(data_file)
        self.support = list(map(lambda x: torch.Tensor(x), data['expected']))

    def test_log_pdf(self):
        log_px_torch = self.g_1.log_pdf(self.test_data[0]).data[0]
        _log_px_np = spr.bernoulli.logpmf(self.test_data.data[0].cpu().numpy(),
                                          p=self.ps_np)
        log_px_np = np.sum(_log_px_np)
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        bs = 2
        log_px_torch = self.g.batch_log_pdf(self.test_data, bs).data.numpy()
        _log_px_np = spr.bernoulli.logpmf(self.test_data.data.cpu().numpy(),
                                          p=self.ps_np)
        log_px_np = [np.sum(_log_px_np[0]), np.sum(_log_px_np[1])]
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_support(self):
        s = list(self.small_g.support())
        v = [torch.equal(x.data, y) for x, y in zip(s, self.support)]
        self.assertTrue(all(v))


class TestLogNormal(TestCase):

    def setUp(self):
        self.mu = Variable(torch.Tensor([1.4]))
        self.sigma = Variable(torch.Tensor([0.4]))
        self.test_data = Variable(torch.Tensor([5.5]))
        self.batch_mu = Variable(torch.Tensor([[1.4], [2.6]]))
        self.batch_sigma = Variable(torch.Tensor([[0.4], [0.5]]))
        self.batch_test_data = Variable(torch.Tensor([[5.5], [6.4]]))
        self.dist = dist.LogNormal(self.mu, self.sigma)
        self.batch_dist = dist.LogNormal(self.batch_mu, self.batch_sigma)
        self.analytic_mean = torch.exp(
            self.mu +
            0.5 *
            torch.pow(
                self.sigma,
                2.0)).data.cpu().numpy()[0]
        var_factor = torch.exp(torch.pow(self.sigma, 2.0)
                               ) - Variable(torch.ones(1))
        self.analytic_var = var_factor.data.cpu().numpy()[
            0] * np.square(self.analytic_mean)
        self.n_samples = 70000

    def test_log_pdf(self):
        log_px_torch = self.dist.log_pdf(self.test_data).data[0]
        log_px_np = spr.lognorm.logpdf(
            self.test_data.data.cpu().numpy(),
            self.sigma.data.cpu().numpy(),
            scale=np.exp(
                self.mu.data.cpu().numpy()))[0]
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_dist.batch_log_pdf(self.batch_test_data).data[0]
        log_px_np = spr.lognorm.logpdf(
            self.test_data.data.cpu().numpy(),
            self.sigma.data.cpu().numpy(),
            scale=np.exp(
                self.mu.data.cpu().numpy()))[0]
        self.assertEqual(log_px_torch[0], log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [self.dist.sample().data.cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(torch_mean, self.analytic_mean, prec=0.1)
        self.assertEqual(torch_var, self.analytic_var, prec=0.1)

    def test_mean_and_var_on_transformed_distribution(self):
        unit_normal = dist.DiagNormal(
            Variable(
                torch.zeros(1)), Variable(
                torch.ones(1)))
        bijector = AffineExp(self.sigma, self.mu)
        trans_dist = TransformedDistribution(unit_normal, bijector)
        torch_samples = [trans_dist.sample().data.cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(torch_mean, self.analytic_mean, prec=0.1)
        self.assertEqual(torch_var, self.analytic_var, prec=0.1)

    def test_log_pdf_on_transformed_distribution(self):
        unit_normal = dist.DiagNormal(
            Variable(
                torch.zeros(1)), Variable(
                torch.ones(1)))
        bijector = AffineExp(self.sigma, self.mu)
        trans_dist = TransformedDistribution(unit_normal, bijector)
        log_px_torch = trans_dist.log_pdf(self.test_data).data[0]
        log_px_np = spr.lognorm.logpdf(
            self.test_data.data.cpu().numpy(),
            self.sigma.data.cpu().numpy(),
            scale=np.exp(
                self.mu.data.cpu().numpy()))[0]
        self.assertEqual(log_px_torch, log_px_np, prec=1e-4)


class TestDelta(TestCase):

    def setUp(self):
        self.v = Variable(torch.Tensor([3]))
        self.vs = Variable(torch.Tensor([[0], [1], [2], [3]]))
        self.test_data = Variable(torch.Tensor([3, 3, 3]))
        self.batch_test_data = Variable(torch.arange(0, 4).unsqueeze(1).expand(4, 3))

        self.dist = dist.Delta(self.v)
        self.batch_dist = dist.Delta(self.vs, batch_size=2)

    def test_log_pdf(self):
        log_px_torch = self.dist.log_pdf(self.test_data).data
        self.assertEqual(torch.sum(log_px_torch), 0)

    def test_batch_log_pdf(self):
        log_px_torch = self.batch_dist.batch_log_pdf(self.batch_test_data).data
        self.assertEqual(torch.sum(log_px_torch), 0)
