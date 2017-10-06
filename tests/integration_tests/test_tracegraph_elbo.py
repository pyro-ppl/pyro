from __future__ import print_function

import pytest
import torch
import torch.optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.transformed_distribution import AffineExp, TransformedDistribution
from pyro.optim.optim import Optimize
from pyro.util import ng_ones, ng_zeros
from tests.common import TestCase

pytestmark = pytest.mark.stage("integration", "integration_batch_2")



def param_mse(name, target):
    return torch.sum(torch.pow(target - pyro.param(name), 2.0)).data.numpy()[0]


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).data.numpy()[0]



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
        self.verbose = True

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 1000)

    @pytest.mark.init(rng_seed=0)
    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 5000)

    def do_elbo_test(self, reparameterized, n_steps):
        if self.verbose:
            print(" - - - - - DO NORMALNORMAL ELBO TEST  [reparameterized = %s] - - - - - " % reparameterized)
        pyro.get_param_store().clear()

        def model():
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.lam0, -0.5),
                                    reparameterized=reparameterized)
            for i, x in enumerate(self.data):
                pyro.observe("obs_%d" % i, dist.diagnormal, x, mu_latent,
                             torch.pow(self.lam, -0.5))
            return mu_latent

        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.334 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.29 * torch.ones(2),
                                   requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            mu_latent = pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q,
                                    reparameterized=reparameterized)
            return mu_latent

	optim = Optimize(model, guide,
			 torch.optim.Adam, {"lr": .0015, "betas": (0.97, 0.999)},
			 loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            optim.step()

            mu_error = param_mse("mu_q", self.analytic_mu_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)
            if k % 250 == 0 and self.verbose:
                print("mu error, log(sigma) error:  %.4f, %.4f" % (mu_error, log_sig_error))

        self.assertEqual(0.0, mu_error, prec=0.03)
        self.assertEqual(0.0, log_sig_error, prec=0.03)


class NormalNormalNormalTests(TestCase):

    def setUp(self):
        # normal-normal-normal; known covariance
        self.lam0 = Variable(torch.Tensor([0.1, 0.1]))  # precision of prior
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
        self.verbose = True

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, True, 5000, 0.02, 0.002, False, False)

    def test_elbo_nonreparameterized_both_baselines(self):
        self.do_elbo_test(False, False, 15000, 0.05, 0.001, use_nn_baseline=True,
                          use_decaying_avg_baseline=True)

    def test_elbo_nonreparameterized_decaying_baseline(self):
        self.do_elbo_test(True, False, 12000, 0.04, 0.0015, use_nn_baseline=False,
                          use_decaying_avg_baseline=True)

    def test_elbo_nonreparameterized_nn_baseline(self):
        self.do_elbo_test(False, True, 12000, 0.04, 0.0015, use_nn_baseline=True,
                          use_decaying_avg_baseline=False)

    def do_elbo_test(self, repa1, repa2, n_steps, prec, lr, use_nn_baseline, use_decaying_avg_baseline):
        if self.verbose:
            print(" - - - - - DO NORMALNORMALNORMAL ELBO TEST - - - - - -")
            print("[reparameterized = %s, %s; nn_baseline = %s, decaying_baseline = %s]" %
                  (repa1, repa2, use_nn_baseline, use_decaying_avg_baseline))
        pyro.get_param_store().clear()

        if use_nn_baseline:

            class VanillaBaselineNN(nn.Module):
                def __init__(self, dim_input, dim_h):
                    super(VanillaBaselineNN, self).__init__()
                    self.lin1 = nn.Linear(dim_input, dim_h)
                    self.lin2 = nn.Linear(dim_h, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    h = self.sigmoid(self.lin1(x))
                    return self.lin2(h)

            mu_prime_baseline = pyro.module("mu_prime_baseline", VanillaBaselineNN(2, 5), scope="baseline")
        else:
            mu_prime_baseline = None

        def model():
            mu_latent_prime = pyro.sample("mu_latent_prime", dist.diagnormal,
                                          self.mu0, torch.pow(self.lam0, -0.5),
                                          reparameterized=repa1)
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    mu_latent_prime, torch.pow(self.lam0, -0.5),
                                    reparameterized=repa2)
            for i, x in enumerate(self.data):
                pyro.observe("obs_%d" % i, dist.diagnormal, x, mu_latent,
                             torch.pow(self.lam, -0.5))
            return mu_latent

        # note that the exact posterior is not mean field!
        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.334 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.29 * torch.ones(2),
                                   requires_grad=True))
            mu_q_prime = pyro.param("mu_q_prime", Variable(torch.Tensor([-0.34, 0.52]),
                                    requires_grad=True))
            kappa_q = pyro.param("kappa_q", Variable(torch.Tensor([0.74]),
                                 requires_grad=True))
            log_sig_q_prime = pyro.param("log_sig_q_prime",
                                         Variable(-0.5 * torch.log(1.2 * self.lam0.data),
                                                  requires_grad=True))
            sig_q, sig_q_prime = torch.exp(log_sig_q), torch.exp(log_sig_q_prime)
            mu_latent_dist = dist.DiagNormal(mu_q, sig_q)
            mu_latent = pyro.sample("mu_latent", mu_latent_dist, reparameterized=repa2,
                                    use_decaying_avg_baseline=use_decaying_avg_baseline)
            mu_latent_prime_dist = dist.DiagNormal(kappa_q.expand_as(mu_latent) * mu_latent + mu_q_prime,
                                                   sig_q_prime)
            pyro.sample("mu_latent_prime",
                        mu_latent_prime_dist,
                        reparameterized=repa1,
                        nn_baseline=mu_prime_baseline,
                        nn_baseline_input=mu_latent,
                        use_decaying_avg_baseline=use_decaying_avg_baseline)

            return mu_latent

        optim = Optimize(model, guide,
                         torch.optim.Adam, {"lr": lr, "betas": (0.97, 0.999)},
                         loss="ELBO", trace_graph=True,
			 auxiliary_optim_constructor=torch.optim.Adam,
			 auxiliary_optim_args={"lr": 5.0 * lr, "betas": (0.90, 0.999)})

        for k in range(n_steps):
            optim.step()

            mu_error = param_mse("mu_q", self.analytic_mu_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)
            mu_prime_error = param_mse("mu_q_prime", 0.5 * self.mu0)
            kappa_error = param_mse("kappa_q", 0.5 * ng_ones(1))
            log_sig_prime_error = param_mse("log_sig_q_prime", -0.5 * torch.log(2.0 * self.lam0))

            if k % 500 == 0 and self.verbose:
                print("errors:  %.4f, %.4f" % (mu_error, log_sig_error), end='')
                print(", %.4f, %.4f" % (mu_prime_error, log_sig_prime_error), end='')
                print(", %.4f" % kappa_error)

        self.assertEqual(0.0, mu_error, prec=prec)
        self.assertEqual(0.0, log_sig_error, prec=prec)
        self.assertEqual(0.0, mu_prime_error, prec=prec)
        self.assertEqual(0.0, log_sig_prime_error, prec=prec)
        self.assertEqual(0.0, kappa_error, prec=prec)


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
        data_sum = self.data[0] + self.data[1] + self.data[2] + self.data[3]
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 - data_sum + \
            Variable(torch.Tensor([self.n_data]))
        # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)
        self.verbose = True

    def test_elbo_nonreparameterized(self):
        if self.verbose:
            print(" - - - - - DO BERNOULLI-BETA ELBO TEST - - - - - ")
        pyro.get_param_store().clear()

        def model():
            p_latent = pyro.sample("p_latent", dist.beta, self.alpha0, self.beta0)
            for i, x in enumerate(self.data):
                pyro.observe("obs_{}".format(i), dist.bernoulli, x,
                             torch.pow(torch.pow(p_latent, 2.0), 0.5))
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     Variable(self.log_alpha_n.data + 0.17, requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    Variable(self.log_beta_n.data - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            p_latent = pyro.sample("p_latent", dist.beta, alpha_q, beta_q,
                                   use_avg_decaying_baseline=True)
            return p_latent

        optim = Optimize(model, guide,
                         torch.optim.Adam, {"lr": 0.0007, "betas": (0.96, 0.999)},
                         loss="ELBO", trace_graph=True)

        for k in range(12000):
            optim.step()

            alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
            beta_error = param_abs_error("beta_q_log", self.log_beta_n)

            if k % 500 == 0 and self.verbose:
                print("alpha_error, beta_error: %.4f, %.4f" % (alpha_error, beta_error))

        self.assertEqual(0.0, alpha_error, prec=0.04)
        self.assertEqual(0.0, beta_error, prec=0.06)


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
        self.verbose = True

    def test_elbo_nonreparameterized(self):
        if self.verbose:
            print(" - - - - - DO POISSON-GAMMA ELBO TEST - - - - - ")
        pyro.get_param_store().clear()

        def model():
            lambda_latent = pyro.sample("lambda_latent", dist.gamma, self.alpha0, self.beta0)
            for i, x in enumerate(self.data):
                pyro.observe("obs_{}".format(i), dist.poisson, x, lambda_latent)
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
            pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q, use_decaying_avg_baseline=True)

        optim = Optimize(model, guide,
                         torch.optim.Adam, {"lr": 0.0007, "betas": (0.95, 0.999)},
                         loss="ELBO", trace_graph=True)

        for k in range(7000):
            optim.step()
            alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
            beta_error = param_abs_error("beta_q_log", self.log_beta_n)
            if k % 500 == 0 and self.verbose:
                print("alpha_q_log_error, beta_q_log_error: %.4f, %.4f" % (alpha_error, beta_error))

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
        self.verbose = True

    def test_elbo_nonreparameterized(self):
        if self.verbose:
            print(" - - - - - DO EXPONENTIAL-GAMMA ELBO TEST - - - - - ")
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
            pyro.sample("lambda_latent", dist.gamma, alpha_q, beta_q, use_decaying_avg_baseline=True)

        optim = Optimize(model, guide,
                         torch.optim.Adam, {"lr": 0.0007, "betas": (0.95, 0.999)},
                         loss="ELBO", trace_graph=True)

        for k in range(8000):
            optim.step()

            alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
            beta_error = param_abs_error("beta_q_log", self.log_beta_n)

            if k % 500 == 0 and self.verbose:
                print("alpha_error, beta_error: %.4f, %.4f" % (alpha_error, beta_error))

        self.assertEqual(0.0, alpha_error, prec=0.03)
        self.assertEqual(0.0, beta_error, prec=0.03)


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
        self.verbose = True

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 7000, 0.95, 0.001)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 7000, 0.95, 0.001)

    def do_elbo_test(self, reparameterized, n_steps, beta1, lr):
        if self.verbose:
            print(" - - - - - DO LOGNORMAL-NORMAL ELBO TEST [repa = %s] - - - - - " % reparameterized)
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
            pyro.sample("mu_latent", dist.diagnormal, mu_q, sigma,
                        reparameterized=reparameterized, use_decaying_avg_baseline=True)

        optim = Optimize(model, guide,
                         torch.optim.Adam, {"lr": lr, "betas": (beta1, 0.999)},
                         loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            optim.step()

            mu_error = param_abs_error("mymodule$$$mu_q_log", self.log_mu_n)
            tau_error = param_abs_error("mymodule$$$tau_q_log", self.log_tau_n)
            if k % 500 == 0 and self.verbose:
                print("mu_error, tau_error = %.4f, %.4f" % (mu_error, tau_error))

        self.assertEqual(0.0, mu_error, prec=0.05)
        self.assertEqual(0.0, tau_error, prec=0.05)

    def test_elbo_with_transformed_distribution(self):
        if self.verbose:
            print(" - - - - - DO LOGNORMAL-NORMAL ELBO TEST [uses TransformedDistribution] - - - - - ")
        pyro.get_param_store().clear()

        def model():
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.tau0, -0.5))
            bijector = AffineExp(torch.pow(self.tau, -0.5), mu_latent)
            x_dist = TransformedDistribution(dist.diagnormal, bijector)
            pyro.observe("obs0", x_dist, self.data[0], ng_zeros(1), ng_ones(1))
            pyro.observe("obs1", x_dist, self.data[1], ng_zeros(1), ng_ones(1))
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

        optim = Optimize(model, guide,
                         torch.optim.Adam, {"lr": .001, "betas": (0.95, 0.999)},
                         loss="ELBO", trace_graph=True)

        for k in range(7000):
            optim.step()

            mu_error = param_abs_error("mu_q_log", self.log_mu_n)
            tau_error = param_abs_error("tau_q_log", self.log_tau_n)

            if k % 500 == 0 and self.verbose:
                print("mu_error, tau_error = %.4f, %.4f" % (mu_error, tau_error))

        self.assertEqual(0.0, mu_error, prec=0.05)
        self.assertEqual(0.0, tau_error, prec=0.05)
