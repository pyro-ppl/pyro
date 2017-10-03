import pytest
import torch
import torch.optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.transformed_distribution import AffineExp, TransformedDistribution
from pyro.infer.kl_qp import KL_QP
from tests.common import TestCase

pytestmark = pytest.mark.init(rng_seed=123)


class NormalNormalTests(TestCase):

    def setUp(self):
        # normal-normal; known covariance
        self.lam0 = Variable(torch.Tensor([0.1, 0.1]))   # precision of prior
        self.mu0 = Variable(torch.Tensor([0.0, 0.5]))   # prior mean
        # known precision of observation noise
        self.lam = Variable(torch.Tensor([6.0, 4.0]))
        self.data = []
        self.data.append(Variable(torch.Tensor([-0.1, 0.3])))
        self.data.append(Variable(torch.Tensor([0.00, 0.4])))
        self.data.append(Variable(torch.Tensor([0.20, 0.5])))
        self.data.append(Variable(torch.Tensor([0.10, 0.7])))
        self.n_data = Variable(torch.Tensor([len(self.data)]))
        self.sum_data = self.data[0] + \
            self.data[1] + self.data[2] + self.data[3]
        self.analytic_lam_n = self.lam0 + \
            self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.sum_data * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)
        self.n_is_samples = 5000
        self.batch_size = 0

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 5000)

    # FIXME
    # def test_elbo_nonreparameterized(self):
    #     self.do_elbo_test(False, 15000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.get_param_store().clear()

        def model():
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.lam0, -0.5))
            pyro.map_data("aaa", self.data, lambda i,
                          x: pyro.observe(
                              "obs_%d" % i, dist.diagnormal,
                              x, mu_latent, torch.pow(self.lam, -0.5)),
                          batch_size=self.batch_size)
            return mu_latent

        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.134 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.09 * torch.ones(2),
                                   requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            dist.diagnormal.reparameterized = reparameterized
            pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q)
            pyro.map_data("aaa", self.data, lambda i, x: None,
                          batch_size=self.batch_size)

        kl_optim = KL_QP(
            model, guide, pyro.optim(
                torch.optim.Adam, {
                    "lr": .001}))
        for k in range(n_steps):
            kl_optim.step()

        mu_error = torch.sum(
            torch.pow(
                self.analytic_mu_n -
                pyro.param("mu_q"),
                2.0))
        log_sig_error = torch.sum(
            torch.pow(
                self.analytic_log_sig_n -
                pyro.param("log_sig_q"),
                2.0))
        self.assertEqual(0.0, mu_error.data.cpu().numpy()[0], prec=0.05)
        self.assertEqual(0.0, log_sig_error.data.cpu().numpy()[0], prec=0.05)

# THIS TEST IS BROKEN BECAUSE OF EXPECTATION/BATCH DIMENSION ISSUES
#     def test_importance_sampling(self):
#   def model():
#       mu_latent = pyro.sample("mu_latent",
#                     DiagNormal(self.mu0, torch.pow(self.lam0, -0.5)))
#       x_dist = DiagNormal(mu_latent, torch.pow(self.lam, -0.5))
#       x = pyro.observe("obs", x_dist, self.data)
#               return mu_latent
#
#   def guide():
#       mu_latent = pyro.sample("mu_latent",
#                   DiagNormal(self.analytic_mu_n, 1.1*torch.pow(self.analytic_lam_n, -0.5)))
#
#         is_infer = ImportanceSampling(model, guide)
#         expected_mean0 = lw_expectation(is_infer, lambda x: x[0], self.n_is_samples)
#         expected_mean1 = lw_expectation(is_infer, lambda x: x[1], self.n_is_samples)
#         expected_mean = Variable(torch.Tensor([expected_mean0.data.cpu().numpy()[0],
#                         expected_mean1.data.cpu().numpy()[0]]))
#         expected_var0  = lw_expectation(is_infer, lambda x: torch.pow(x[0]-expected_mean0,2.0),
#                                        self.n_is_samples).data.cpu().numpy()[0]
#         expected_var1  = lw_expectation(is_infer, lambda x: torch.pow(x[1]-expected_mean1,2.0),
#                                        self.n_is_samples).data.cpu().numpy()[0]
#         expected_var = (expected_var0 + expected_var1)
#         analytic_var = torch.sum(torch.pow(self.analytic_lam_n,-1.0)).data.cpu().numpy()[0]
#
#         mu_error  = torch.sum(torch.pow(expected_mean-self.analytic_mu_n,2.0)).data.cpu().numpy()[0]
#         var_error = analytic_var - expected_var
#   self.assertEqual(0.0, mu_error, prec=0.005)
#   self.assertEqual(0.0, var_error, prec=0.010)


class TestFixedModelGuide(TestCase):
    def setUp(self):
        self.data = Variable(torch.Tensor([2.0]))
        self.alpha_q_log_0 = 0.17 * torch.ones(1)
        self.beta_q_log_0 = 0.19 * torch.ones(1)
        self.alpha_p_log_0 = 0.11 * torch.ones(1)
        self.beta_p_log_0 = 0.13 * torch.ones(1)

    def do_test_fixedness(self, model_fixed, guide_fixed):
        pyro.get_param_store().clear()

        def model():
            alpha_p_log = pyro.param(
                "alpha_p_log", Variable(
                    self.alpha_p_log_0, requires_grad=True))
            beta_p_log = pyro.param(
                "beta_p_log", Variable(
                    self.beta_p_log_0, requires_grad=True))
            alpha_p, beta_p = torch.exp(alpha_p_log), torch.exp(beta_p_log)
            lambda_latent = pyro.sample("lambda_latent", dist.gamma, alpha_p, beta_p)
            pyro.observe("obs", dist.poisson, self.data, lambda_latent)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log", Variable(
                    self.alpha_q_log_0, requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log", Variable(
                    self.beta_q_log_0, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q)

        kl_optim = KL_QP(model, guide, pyro.optim(torch.optim.Adam, {"lr": .001}),
                         model_fixed=model_fixed, guide_fixed=guide_fixed)
        for _ in range(10):
            kl_optim.step()

        model_unchanged = (torch.equal(pyro.param("alpha_p_log").data, self.alpha_p_log_0)) and\
                          (torch.equal(pyro.param("beta_p_log").data, self.beta_p_log_0))
        guide_unchanged = (torch.equal(pyro.param("alpha_q_log").data, self.alpha_q_log_0)) and\
                          (torch.equal(pyro.param("beta_q_log").data, self.beta_q_log_0))
        bad = (model_fixed and (not model_unchanged)) or (guide_fixed and (not guide_unchanged))
        return (not bad)

    def test_model_fixed(self):
        assert self.do_test_fixedness(model_fixed=True, guide_fixed=False)

    def test_guide_fixed(self):
        assert self.do_test_fixedness(model_fixed=False, guide_fixed=True)

    def test_guide_and_model_fixed(self):
        assert self.do_test_fixedness(model_fixed=True, guide_fixed=True)


class PoissonGammaTests(TestCase):
    def setUp(self):
        # poisson-gamma model
        # gamma prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        # gamma prior hyperparameter
        self.beta0 = Variable(torch.Tensor([1.0]))
        self.data = []
        self.data.append(Variable(torch.Tensor([1.0])))
        self.data.append(Variable(torch.Tensor([2.0])))
        self.data.append(Variable(torch.Tensor([3.0])))
        self.n_data = len(self.data)
        sum_data = self.data[0] + self.data[1] + self.data[2]
        self.alpha_n = self.alpha0 + sum_data  # posterior alpha
        self.beta_n = self.beta0 + \
            Variable(torch.Tensor([self.n_data]))  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test_elbo_nonreparameterized(self):
        pyro.get_param_store().clear()

        def model():
            lambda_latent = pyro.sample("lambda_latent", dist.gamma, self.alpha0, self.beta0)
            pyro.map_data("aaa",
                          self.data, lambda i, x: pyro.observe(
                              "obs_{}".format(i), dist.poisson, x, lambda_latent), batch_size=3)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log",
                Variable(
                    self.log_alpha_n.data +
                    0.17,
                    requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log",
                Variable(
                    self.log_beta_n.data -
                    0.143,
                    requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q)
            pyro.map_data("aaa", self.data, lambda i, x: None, batch_size=3)

        kl_optim = KL_QP(
            model, guide, pyro.optim(
                torch.optim.Adam, {
                    "lr": .0002, "betas": (
                        0.97, 0.999)}))
        for k in range(25000):
            kl_optim.step()
#            if k%1000==0:
#                 print "alpha_q", torch.exp(pyro.param("alpha_q_log")).data.numpy()[0]
#                 print "beta_q", torch.exp(pyro.param("beta_q_log")).data.numpy()[0]
#
#         print "alpha_n", self.alpha_n.data.numpy()[0]
#         print "beta_n", self.beta_n.data.numpy()[0]
#         print "alpha_0", self.alpha0.data.numpy()[0]
#         print "beta_0", self.beta0.data.numpy()[0]

        alpha_error = torch.abs(
            pyro.param("alpha_q_log") -
            self.log_alpha_n).data.cpu().numpy()[0]
        beta_error = torch.abs(
            pyro.param("beta_q_log") -
            self.log_beta_n).data.cpu().numpy()[0]
        self.assertEqual(0.0, alpha_error, prec=0.08)
        self.assertEqual(0.0, beta_error, prec=0.08)


class ExponentialGammaTests(TestCase):
    def setUp(self):
        # exponential-gamma model
        # gamma prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        # gamma prior hyperparameter
        self.beta0 = Variable(torch.Tensor([1.0]))
        self.n_data = 2
        self.data = Variable(torch.Tensor([3.0, 2.0]))  # two observations
        self.alpha_n = self.alpha0 + \
            Variable(torch.Tensor([self.n_data]))  # posterior alpha
        self.beta_n = self.beta0 + torch.sum(self.data)  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test_elbo_nonreparameterized(self):
        pyro.get_param_store().clear()

        def model():
            lambda_latent = pyro.sample("lambda_latent", dist.gamma, self.alpha0, self.beta0)
            pyro.observe("obs0", dist.exponential, self.data[0], lambda_latent)
            pyro.observe("obs1", dist.exponential, self.data[1], lambda_latent)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log",
                Variable(self.log_alpha_n.data + 0.17, requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log",
                Variable(self.log_beta_n.data - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q)

        kl_optim = KL_QP(
            model, guide, pyro.optim(
                torch.optim.Adam, {
                    "lr": .0003, "betas": (
                        0.97, 0.999)}))
        for k in range(10001):
            kl_optim.step()

        alpha_error = torch.abs(
            pyro.param("alpha_q_log") -
            self.log_alpha_n).data.cpu().numpy()[0]
        beta_error = torch.abs(
            pyro.param("beta_q_log") -
            self.log_beta_n).data.cpu().numpy()[0]
        # print "alpha_error", alpha_error
        # print "beta_error", beta_error
        self.assertEqual(0.0, alpha_error, prec=0.08)
        self.assertEqual(0.0, beta_error, prec=0.08)


class BernoulliBetaTests(TestCase):
    def setUp(self):
        # bernoulli-beta model
        # beta prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        self.beta0 = Variable(torch.Tensor([1.0]))  # beta prior hyperparameter
        self.data = []
        self.data.append(Variable(torch.Tensor([0.0])))
        self.data.append(Variable(torch.Tensor([1.0])))
        self.data.append(Variable(torch.Tensor([1.0])))
        self.data.append(Variable(torch.Tensor([1.0])))
        self.n_data = len(self.data)
        self.batch_size = 0
        self.n_steps = 6001
        data_sum = self.data[0] + self.data[1] + self.data[2] + self.data[3]
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 - data_sum + \
            Variable(torch.Tensor([self.n_data]))
        # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test_elbo_nonreparameterized(self):
        pyro.get_param_store().clear()

        def model():
            p_latent = pyro.sample("p_latent", dist.beta, self.alpha0, self.beta0)
            pyro.map_data("aaa",
                          self.data, lambda i, x: pyro.observe(
                              "obs_{}".format(i), dist.bernoulli, x, p_latent),
                          batch_size=self.batch_size)
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     Variable(self.log_alpha_n.data + 0.17, requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    Variable(self.log_beta_n.data - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("p_latent", dist.beta, alpha_q, beta_q)
            pyro.map_data("aaa", self.data, lambda i, x: None, batch_size=self.batch_size)

        kl_optim = KL_QP(model, guide, pyro.optim(torch.optim.Adam,
                                                  {"lr": .001, "betas": (0.97, 0.999)}))
        for k in range(self.n_steps):
            kl_optim.step()
#             if k%1000==0:
#                 print "alpha_q", torch.exp(pyro.param("alpha_q_log")).data.numpy()[0]
#                 print "beta_q", torch.exp(pyro.param("beta_q_log")).data.numpy()[0]
#
#         print "alpha_n", self.alpha_n.data.numpy()[0]
#         print "beta_n", self.beta_n.data.numpy()[0]
#         print "alpha_0", self.alpha0.data.numpy()[0]
#         print "beta_0", self.beta0.data.numpy()[0]

        alpha_error = torch.abs(
            pyro.param("alpha_q_log") -
            self.log_alpha_n).data.cpu().numpy()[0]
        beta_error = torch.abs(
            pyro.param("beta_q_log") -
            self.log_beta_n).data.cpu().numpy()[0]
        self.assertEqual(0.0, alpha_error, prec=0.08)
        self.assertEqual(0.0, beta_error, prec=0.08)


class LogNormalNormalGuide(nn.Module):
    def __init__(self, mu_q_log_init, tau_q_log_init):
        super(LogNormalNormalGuide, self).__init__()
        self.mu_q_log = Parameter(mu_q_log_init)
        self.tau_q_log = Parameter(tau_q_log_init)


class LogNormalNormalTests(TestCase):
    def setUp(self):
        # lognormal-normal model
        # putting some of the parameters inside of a torch module to
        # make sure that that functionality is ok (XXX: do this somewhere else in the future)
        self.mu0 = Variable(torch.Tensor([[1.0]]))  # normal prior hyperparameter
        # normal prior hyperparameter
        self.tau0 = Variable(torch.Tensor([[1.0]]))
        # known precision for observation likelihood
        self.tau = Variable(torch.Tensor([[2.5]]))
        self.n_data = 2
        self.data = Variable(torch.Tensor([[1.5], [2.2]]))  # two observations
        self.tau_n = self.tau0 + \
            Variable(torch.Tensor([self.n_data])) * self.tau  # posterior tau
        mu_numerator = self.mu0 * self.tau0 + \
            self.tau * torch.sum(torch.log(self.data))
        self.mu_n = mu_numerator / self.tau_n  # posterior mu
        self.log_mu_n = torch.log(self.mu_n)
        self.log_tau_n = torch.log(self.tau_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 12000)

    # FIXME
    # def test_elbo_nonreparameterized(self):
    #     self.do_elbo_test(False, 15000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.get_param_store().clear()
        pt_guide = LogNormalNormalGuide(self.log_mu_n.data + 0.17,
                                        self.log_tau_n.data - 0.143)

        def model():
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.tau0, -0.5))
            sigma = torch.pow(self.tau, -0.5)
            pyro.observe("obs0", dist.lognormal, self.data[0], mu_latent, sigma)
            pyro.observe("obs1", dist.lognormal, self.data[1], mu_latent, sigma)
            return mu_latent

        def guide():
            pyro.module("mymodule", pt_guide)
            mu_q, tau_q = torch.exp(pt_guide.mu_q_log), torch.exp(pt_guide.tau_q_log)
            sigma = torch.pow(tau_q, -0.5)
            dist.diagnormal.reparameterized = reparameterized
            pyro.sample("mu_latent", dist.diagnormal, mu_q, sigma)

        kl_optim = KL_QP(model, guide, pyro.optim(torch.optim.Adam,
                                                  {"lr": .0005, "betas": (0.96, 0.999)}))
        for k in range(n_steps):
            kl_optim.step()

        mu_error = torch.abs(
            pyro.param("mymodule$$$mu_q_log") -
            self.log_mu_n).data.cpu().numpy()[0][0]
        tau_error = torch.abs(
            pyro.param("mymodule$$$tau_q_log") -
            self.log_tau_n).data.cpu().numpy()[0][0]
        # print "mu_error", mu_error
        # print "tau_error", tau_error
        self.assertEqual(0.0, mu_error, prec=0.07)
        self.assertEqual(0.0, tau_error, prec=0.07)

    def test_elbo_with_transformed_distribution(self):
        pyro.get_param_store().clear()

        def model():
            zero = Variable(torch.zeros(1, 1))
            one = Variable(torch.ones(1, 1))
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.tau0, -0.5))
            bijector = AffineExp(torch.pow(self.tau, -0.5), mu_latent)
            x_dist = TransformedDistribution(dist.diagnormal, bijector)
            pyro.observe("obs0", x_dist, self.data[0], zero, one)
            pyro.observe("obs1", x_dist, self.data[1], zero, one)
            return mu_latent

        def guide():
            mu_q_log = pyro.param(
                "mu_q_log",
                Variable(
                    self.log_mu_n.data +
                    0.17,
                    requires_grad=True))
            tau_q_log = pyro.param("tau_q_log", Variable(self.log_tau_n.data - 0.143,
                                                         requires_grad=True))
            mu_q, tau_q = torch.exp(mu_q_log), torch.exp(tau_q_log)
            pyro.sample("mu_latent", dist.diagnormal, mu_q, torch.pow(tau_q, -0.5))

        kl_optim = KL_QP(model, guide, pyro.optim(torch.optim.Adam,
                                                  {"lr": .0005, "betas": (0.96, 0.999)}))
        for k in range(12001):
            kl_optim.step()

        mu_error = torch.abs(
            pyro.param("mu_q_log") -
            self.log_mu_n).data.cpu().numpy()[0][0]
        tau_error = torch.abs(
            pyro.param("tau_q_log") -
            self.log_tau_n).data.cpu().numpy()[0][0]
        self.assertEqual(0.0, mu_error, prec=0.05)
        self.assertEqual(0.0, tau_error, prec=0.05)
