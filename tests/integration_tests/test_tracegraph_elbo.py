from __future__ import absolute_import, division, print_function

import logging
from unittest import TestCase

import numpy as np
import pytest
import torch
from torch import nn as nn
from torch.autograd import Variable, variable
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.distributions.testing import fakes
from pyro.distributions import TransformedDistribution
from pyro.infer import SVI
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal
from tests.distributions.test_transformed_distribution import AffineExp

pytestmark = pytest.mark.stage("integration", "integration_batch_2")
logger = logging.getLogger(__name__)


def param_mse(name, target):
    return torch.sum(torch.pow(target - pyro.param(name), 2.0)).item()


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).item()


class NormalNormalTests(TestCase):

    def setUp(self):
        # normal-normal; known covariance
        self.lam0 = variable([0.1, 0.1])   # precision of prior
        self.mu0 = variable([0.0, 0.5])   # prior mean
        # known precision of observation noise
        self.lam = variable([6.0, 4.0])
        self.data = []
        self.data.append(variable([-0.1, 0.3]))
        self.data.append(variable([0.00, 0.4]))
        self.data.append(variable([0.20, 0.5]))
        self.data.append(variable([0.10, 0.7]))
        self.n_data = variable(len(self.data))
        self.sum_data = self.data[0] + \
            self.data[1] + self.data[2] + self.data[3]
        self.analytic_lam_n = self.lam0 + \
            self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.sum_data * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 1500, 0.02)

    @pytest.mark.init(rng_seed=0)
    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 5000, 0.05)

    def do_elbo_test(self, reparameterized, n_steps, prec):
        logger.info(" - - - - - DO NORMALNORMAL ELBO TEST  [reparameterized = %s] - - - - - " % reparameterized)
        pyro.clear_param_store()
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

        def model():
            with pyro.iarange("iarange", 2):
                mu_latent = pyro.sample("mu_latent", Normal(self.mu0, torch.pow(self.lam0, -0.5)))
                for i, x in enumerate(self.data):
                    pyro.sample("obs_%d" % i,
                                dist.Normal(mu_latent, torch.pow(self.lam, -0.5)),
                                obs=x)
            return mu_latent

        def guide():
            mu_q = pyro.param("mu_q", variable(self.analytic_mu_n.expand(2) + 0.334, requires_grad=True))
            log_sig_q = pyro.param("log_sig_q",
                                   variable(self.analytic_log_sig_n.expand(2) - 0.29, requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            with pyro.iarange("iarange", 2):
                mu_latent = pyro.sample("mu_latent", Normal(mu_q, sig_q))
            return mu_latent

        adam = optim.Adam({"lr": .0015, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            svi.step()

            mu_error = param_mse("mu_q", self.analytic_mu_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)
            if k % 250 == 0:
                logger.debug("mu error, log(sigma) error:  %.4f, %.4f" % (mu_error, log_sig_error))

        assert_equal(0.0, mu_error, prec=prec)
        assert_equal(0.0, log_sig_error, prec=prec)


class NormalNormalNormalTests(TestCase):

    def setUp(self):
        # normal-normal-normal; known covariance
        self.lam0 = variable([0.1, 0.1])  # precision of prior
        self.mu0 = variable([0.0, 0.5])   # prior mean
        # known precision of observation noise
        self.lam = variable([6.0, 4.0])
        self.data = variable([[-0.1, 0.3],
                              [0.00, 0.4],
                              [0.20, 0.5],
                              [0.10, 0.7]])
        self.analytic_lam_n = self.lam0 + len(self.data) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.data.sum(0) * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, True, 3000, 0.02, 0.002, False, False)

    def test_elbo_nonreparameterized_both_baselines(self):
        self.do_elbo_test(False, False, 3000, 0.04, 0.001, use_nn_baseline=True,
                          use_decaying_avg_baseline=True)

    def test_elbo_nonreparameterized_decaying_baseline(self):
        self.do_elbo_test(True, False, 4000, 0.04, 0.0015, use_nn_baseline=False,
                          use_decaying_avg_baseline=True)

    def test_elbo_nonreparameterized_nn_baseline(self):
        self.do_elbo_test(False, True, 4000, 0.04, 0.0015, use_nn_baseline=True,
                          use_decaying_avg_baseline=False)

    def do_elbo_test(self, repa1, repa2, n_steps, prec, lr, use_nn_baseline, use_decaying_avg_baseline):
        logger.info(" - - - - - DO NORMALNORMALNORMAL ELBO TEST - - - - - -")
        logger.info("[reparameterized = %s, %s; nn_baseline = %s, decaying_baseline = %s]" %
                    (repa1, repa2, use_nn_baseline, use_decaying_avg_baseline))
        pyro.clear_param_store()
        Normal1 = dist.Normal if repa1 else fakes.NonreparameterizedNormal
        Normal2 = dist.Normal if repa2 else fakes.NonreparameterizedNormal

        if use_nn_baseline:

            class VanillaBaselineNN(nn.Module):
                def __init__(self, dim_input, dim_h):
                    super(VanillaBaselineNN, self).__init__()
                    self.lin1 = nn.Linear(dim_input, dim_h)
                    self.lin2 = nn.Linear(dim_h, 2)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    h = self.sigmoid(self.lin1(x))
                    return self.lin2(h)

            mu_prime_baseline = pyro.module("mu_prime_baseline", VanillaBaselineNN(2, 5), tags="baseline")
        else:
            mu_prime_baseline = None

        def model():
            with pyro.iarange("iarange", 2):
                mu_latent_prime = pyro.sample("mu_latent_prime", Normal1(self.mu0, torch.pow(self.lam0, -0.5)))
                mu_latent = pyro.sample("mu_latent", Normal2(mu_latent_prime, torch.pow(self.lam0, -0.5)))
                with pyro.iarange("data", len(self.data)):
                    pyro.sample("obs",
                                dist.Normal(mu_latent, torch.pow(self.lam, -0.5))
                                    .reshape(sample_shape=self.data.shape[:1]),
                                obs=self.data)
            return mu_latent

        # note that the exact posterior is not mean field!
        def guide():
            mu_q = pyro.param("mu_q", variable(self.analytic_mu_n.expand(2) + 0.334, requires_grad=True))
            log_sig_q = pyro.param("log_sig_q",
                                   variable(self.analytic_log_sig_n.expand(2) - 0.29, requires_grad=True))
            mu_q_prime = pyro.param("mu_q_prime",
                                    variable([-0.34, 0.52], requires_grad=True))
            kappa_q = pyro.param("kappa_q", variable([0.74],
                                 requires_grad=True))
            log_sig_q_prime = pyro.param("log_sig_q_prime",
                                         variable(-0.5 * torch.log(1.2 * self.lam0), requires_grad=True))
            sig_q, sig_q_prime = torch.exp(log_sig_q), torch.exp(log_sig_q_prime)
            with pyro.iarange("iarange", 2):
                mu_latent = pyro.sample("mu_latent", Normal2(mu_q, sig_q),
                                        infer=dict(baseline=dict(use_decaying_avg_baseline=use_decaying_avg_baseline)))
                pyro.sample("mu_latent_prime",
                            Normal1(kappa_q.expand_as(mu_latent) * mu_latent + mu_q_prime, sig_q_prime),
                            infer=dict(baseline=dict(nn_baseline=mu_prime_baseline,
                                                     nn_baseline_input=mu_latent,
                                                     use_decaying_avg_baseline=use_decaying_avg_baseline)))
                with pyro.iarange("data", len(self.data)):
                    pass

            return mu_latent

        adam = optim.Adam({"lr": .0015, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            svi.step()

            mu_error = param_mse("mu_q", self.analytic_mu_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)
            mu_prime_error = param_mse("mu_q_prime", 0.5 * self.mu0)
            kappa_error = param_mse("kappa_q", 0.5 * ng_ones(1))
            log_sig_prime_error = param_mse("log_sig_q_prime", -0.5 * torch.log(2.0 * self.lam0))

            if k % 500 == 0:
                logger.debug("errors:  %.4f, %.4f" % (mu_error, log_sig_error))
                logger.debug(", %.4f, %.4f" % (mu_prime_error, log_sig_prime_error))
                logger.debug(", %.4f" % kappa_error)

        assert_equal(0.0, mu_error, prec=prec)
        assert_equal(0.0, log_sig_error, prec=prec)
        assert_equal(0.0, mu_prime_error, prec=prec)
        assert_equal(0.0, log_sig_prime_error, prec=prec)
        assert_equal(0.0, kappa_error, prec=prec)


class BernoulliBetaTests(TestCase):
    def setUp(self):
        # bernoulli-beta model
        # beta prior hyperparameter
        self.alpha0 = variable(1.0)
        self.beta0 = variable(1.0)  # beta prior hyperparameter
        self.data = variable([0.0, 1.0, 1.0, 1.0])
        self.n_data = len(self.data)
        data_sum = self.data.sum()
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 - data_sum + variable(self.n_data)  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 3000, 0.92, 0.0007)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 3000, 0.95, 0.0007)

    def do_elbo_test(self, reparameterized, n_steps, beta1, lr):
        logger.info(" - - - - - DO BETA-BERNOULLI ELBO TEST [repa = %s] - - - - - " % reparameterized)
        pyro.clear_param_store()
        Beta = dist.Beta if reparameterized else fakes.NonreparameterizedBeta

        def model():
            p_latent = pyro.sample("p_latent", Beta(self.alpha0, self.beta0))
            with pyro.iarange("data", len(self.data)):
                pyro.sample("obs", dist.Bernoulli(p_latent), obs=self.data)
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     variable(self.log_alpha_n + 0.17, requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    variable(self.log_beta_n - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            p_latent = pyro.sample("p_latent", Beta(alpha_q, beta_q),
                                   infer=dict(baseline=dict(use_decaying_avg_baseline=True)))
            with pyro.iarange("data", len(self.data)):
                pass
            return p_latent

        adam = optim.Adam({"lr": lr, "betas": (beta1, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            svi.step()
            alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
            beta_error = param_abs_error("beta_q_log", self.log_beta_n)
            if k % 500 == 0:
                logger.debug("alpha_error, beta_error: %.4f, %.4f" % (alpha_error, beta_error))

        assert_equal(0.0, alpha_error, prec=0.03)
        assert_equal(0.0, beta_error, prec=0.04)


class ExponentialGammaTests(TestCase):
    def setUp(self):
        # exponential-gamma model
        # gamma prior hyperparameter
        self.alpha0 = variable(1.0)
        # gamma prior hyperparameter
        self.beta0 = variable(1.0)
        self.n_data = 2
        self.data = variable([3.0, 2.0])  # two observations
        self.alpha_n = self.alpha0 + self.n_data  # posterior alpha
        self.beta_n = self.beta0 + self.data.sum()  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 8000, 0.90, 0.0007)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 8000, 0.95, 0.0007)

    def do_elbo_test(self, reparameterized, n_steps, beta1, lr):
        logger.info(" - - - - - DO EXPONENTIAL-GAMMA ELBO TEST [repa = %s] - - - - - " % reparameterized)
        pyro.clear_param_store()
        Gamma = dist.Gamma if reparameterized else fakes.NonreparameterizedGamma

        def model():
            lambda_latent = pyro.sample("lambda_latent", Gamma(self.alpha0, self.beta0))
            with pyro.iarange("data", len(self.data)):
                pyro.sample("obs", dist.Exponential(lambda_latent), obs=self.data)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log",
                variable(self.log_alpha_n + 0.17, requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log",
                variable(self.log_beta_n - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", Gamma(alpha_q, beta_q),
                        infer=dict(baseline=dict(use_decaying_avg_baseline=True)))
            with pyro.iarange("data", len(self.data)):
                pass

        adam = optim.Adam({"lr": lr, "betas": (beta1, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            svi.step()
            alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
            beta_error = param_abs_error("beta_q_log", self.log_beta_n)
            if k % 500 == 0:
                logger.debug("alpha_error, beta_error: %.4f, %.4f" % (alpha_error, beta_error))

        assert_equal(0.0, alpha_error, prec=0.04)
        assert_equal(0.0, beta_error, prec=0.04)


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
        self.mu0 = variable(1.0)  # normal prior hyperparameter
        # normal prior hyperparameter
        self.tau0 = variable(1.0)
        # known precision for observation likelihood
        self.tau = variable(2.5)
        self.data = variable([1.5, 2.2])  # two observations
        self.tau_n = self.tau0 + len(self.data) * self.tau  # posterior tau
        mu_numerator = self.mu0 * self.tau0 + self.tau * torch.log(self.data).sum(0)
        self.mu_n = mu_numerator / self.tau_n  # posterior mu
        self.log_mu_n = torch.log(self.mu_n)
        self.log_tau_n = torch.log(self.tau_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 7000, 0.95, 0.001)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 7000, 0.95, 0.001)

    def do_elbo_test(self, reparameterized, n_steps, beta1, lr):
        logger.info(" - - - - - DO LOGNORMAL-NORMAL ELBO TEST [repa = %s] - - - - - " % reparameterized)
        pyro.clear_param_store()
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
        pt_guide = LogNormalNormalGuide(self.log_mu_n.data + 0.17,
                                        self.log_tau_n.data - 0.143)

        def model():
            mu_latent = pyro.sample("mu_latent", dist.Normal(self.mu0, torch.pow(self.tau0, -0.5)))
            sigma = torch.pow(self.tau, -0.5)
            with pyro.iarange("data", len(self.data)):
                pyro.sample("obs", dist.LogNormal(mu_latent, sigma), obs=self.data)
            return mu_latent

        def guide():
            pyro.module("mymodule", pt_guide)
            mu_q = torch.exp(pt_guide.mu_q_log).squeeze()
            tau_q = torch.exp(pt_guide.tau_q_log).squeeze()
            sigma = torch.pow(tau_q, -0.5)
            pyro.sample("mu_latent", Normal(mu_q, sigma),
                        infer=dict(baseline=dict(use_decaying_avg_baseline=True)))
            with pyro.iarange("data", len(self.data)):
                pass

        adam = optim.Adam({"lr": lr, "betas": (beta1, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            svi.step()
            mu_error = param_abs_error("mymodule$$$mu_q_log", self.log_mu_n)
            tau_error = param_abs_error("mymodule$$$tau_q_log", self.log_tau_n)
            if k % 500 == 0:
                logger.debug("mu_error, tau_error = %.4f, %.4f" % (mu_error, tau_error))

        assert_equal(0.0, mu_error, prec=0.05)
        assert_equal(0.0, tau_error, prec=0.05)

    def test_elbo_with_transformed_distribution(self):
        logger.info(" - - - - - DO LOGNORMAL-NORMAL ELBO TEST [uses TransformedDistribution] - - - - - ")
        pyro.clear_param_store()

        def model():
            mu_latent = pyro.sample("mu_latent", dist.Normal(self.mu0, torch.pow(self.tau0, -0.5)))
            bijector = AffineExp(torch.pow(self.tau, -0.5), mu_latent)
            x_dist = TransformedDistribution(dist.Normal(0, 1), bijector)
            pyro.sample("obs0", x_dist, obs=self.data[0])
            pyro.sample("obs1", x_dist, obs=self.data[1])
            return mu_latent

        def guide():
            mu_q_log = pyro.param("mu_q_log",
                                  variable(self.log_mu_n + 0.17, requires_grad=True))
            tau_q_log = pyro.param("tau_q_log",
                                   variable(self.log_tau_n - 0.143, requires_grad=True))
            mu_q, tau_q = torch.exp(mu_q_log), torch.exp(tau_q_log)
            pyro.sample("mu_latent", dist.Normal(mu_q, torch.pow(tau_q, -0.5)))

        adam = optim.Adam({"lr": 0.001, "betas": (0.95, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for k in range(7000):
            svi.step()
            mu_error = param_abs_error("mu_q_log", self.log_mu_n)
            tau_error = param_abs_error("tau_q_log", self.log_tau_n)
            if k % 500 == 0:
                logger.debug("mu_error, tau_error = %.4f, %.4f" % (mu_error, tau_error))

        assert_equal(0.0, mu_error, prec=0.05)
        assert_equal(0.0, tau_error, prec=0.05)


@pytest.mark.init(rng_seed=0)
@pytest.mark.stage("integration", "integration_batch_1")
class RaoBlackwellizationTests(TestCase):
    def setUp(self):
        # normal-normal; known covariance
        self.lam0 = variable([0.1, 0.1])   # precision of prior
        self.mu0 = variable([0.0, 0.5])   # prior mean
        # known precision of observation noise
        self.lam = variable([6.0, 4.0])
        self.n_outer = 3
        self.n_inner = 3
        self.n_data = variable(self.n_outer * self.n_inner)
        self.data = []
        self.sum_data = ng_zeros(2)
        for _out in range(self.n_outer):
            data_in = []
            for _in in range(self.n_inner):
                data_in.append(variable([-0.1, 0.3]) + variable(torch.Size([2])).normal_() / self.lam.sqrt())
                self.sum_data += data_in[-1]
            self.data.append(data_in)
        self.analytic_lam_n = self.lam0 + self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.sum_data * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)

    # this tests rao-blackwellization in elbo for nested iranges
    def test_nested_irange_in_elbo(self, n_steps=4000):
        pyro.clear_param_store()

        def model():
            mu_latent = pyro.sample("mu_latent",
                                    fakes.NonreparameterizedNormal(self.mu0,
                                                                   torch.pow(self.lam0,
                                                                             -0.5)).reshape(extra_event_dims=1))
            for i in pyro.irange("outer", self.n_outer):
                for j in pyro.irange("inner", self.n_inner):
                    pyro.sample("obs_%d_%d" % (i, j),
                                dist.Normal(mu_latent, torch.pow(self.lam, -0.5)).reshape(extra_event_dims=1),
                                obs=self.data[i][j])

        def guide():
            mu_q = pyro.param("mu_q", variable(self.analytic_mu_n.expand(2) + 0.234, requires_grad=True))
            log_sig_q = pyro.param("log_sig_q",
                                   variable(self.analytic_log_sig_n.expand(2) - 0.27, requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            pyro.sample("mu_latent", fakes.NonreparameterizedNormal(mu_q, sig_q).reshape(extra_event_dims=1),
                        infer=dict(baseline=dict(use_decaying_avg_baseline=True)))

            for i in pyro.irange("outer", self.n_outer):
                for j in pyro.irange("inner", self.n_inner):
                    pass

        guide_trace = pyro.poutine.trace(guide, graph_type="dense").get_trace()
        model_trace = pyro.poutine.trace(pyro.poutine.replay(model, guide_trace),
                                         graph_type="dense").get_trace()
        assert len(model_trace.edges()) == 27
        assert len(model_trace.nodes()) == 16
        assert len(guide_trace.edges()) == 0
        assert len(guide_trace.nodes()) == 9

        adam = optim.Adam({"lr": 0.0008, "betas": (0.96, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            svi.step()
            mu_error = param_mse("mu_q", self.analytic_mu_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)
            if k % 500 == 0:
                logger.debug("mu error, log(sigma) error:  %.4f, %.4f" % (mu_error, log_sig_error))

        assert_equal(0.0, mu_error, prec=0.04)
        assert_equal(0.0, log_sig_error, prec=0.04)

    # this tests rao-blackwellization and baselines for iarange
    # inside of an irange with superfluous random variables to complexify the
    # graph structure and introduce additional baselines
    def test_iarange_in_elbo_with_superfluous_rvs(self):
        self._test_iarange_in_elbo(n_superfluous_top=1, n_superfluous_bottom=1, n_steps=5000)

    def _test_iarange_in_elbo(self, n_superfluous_top, n_superfluous_bottom, n_steps):
        pyro.clear_param_store()
        self.data_tensor = Variable(torch.zeros(9, 2))
        for _out in range(self.n_outer):
            for _in in range(self.n_inner):
                self.data_tensor[3 * _out + _in, :] = self.data[_out][_in]
        self.data_as_list = [self.data_tensor[0:4, :], self.data_tensor[4:7, :],
                             self.data_tensor[7:9, :]]

        def model():
            mu_latent = pyro.sample("mu_latent",
                                    fakes.NonreparameterizedNormal(self.mu0, torch.pow(self.lam0, -0.5))
                                         .reshape(extra_event_dims=1))

            for i in pyro.irange("outer", 3):
                x_i = self.data_as_list[i]
                with pyro.iarange("inner_%d" % i, x_i.size(0)):
                    for k in range(n_superfluous_top):
                        z_i_k = pyro.sample("z_%d_%d" % (i, k),
                                            fakes.NonreparameterizedNormal(0, 1).reshape(sample_shape=[4 - i]))
                        assert z_i_k.shape == (4 - i,)
                    obs_i = pyro.sample("obs_%d" % i, dist.Normal(mu_latent, torch.pow(self.lam, -0.5))
                                                          .reshape(extra_event_dims=1), obs=x_i)
                    assert obs_i.shape == (4 - i, 2)
                    for k in range(n_superfluous_top, n_superfluous_top + n_superfluous_bottom):
                        z_i_k = pyro.sample("z_%d_%d" % (i, k),
                                            fakes.NonreparameterizedNormal(0, 1).reshape(sample_shape=[4 - i]))
                        assert z_i_k.shape == (4 - i,)

        pt_mu_baseline = torch.nn.Linear(1, 1)
        pt_superfluous_baselines = []
        for k in range(n_superfluous_top + n_superfluous_bottom):
            pt_superfluous_baselines.extend([torch.nn.Linear(2, 4), torch.nn.Linear(2, 3),
                                             torch.nn.Linear(2, 2)])

        def guide():
            mu_q = pyro.param("mu_q", variable(self.analytic_mu_n.expand(2) + 0.094, requires_grad=True))
            log_sig_q = pyro.param("log_sig_q",
                                   variable(self.analytic_log_sig_n.expand(2) - 0.07, requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            trivial_baseline = pyro.module("mu_baseline", pt_mu_baseline, tags="baseline")
            baseline_value = trivial_baseline(ng_ones(1)).squeeze()
            mu_latent = pyro.sample("mu_latent",
                                    fakes.NonreparameterizedNormal(mu_q, sig_q).reshape(extra_event_dims=1),
                                    infer=dict(baseline=dict(baseline_value=baseline_value)))

            for i in pyro.irange("outer", 3):
                with pyro.iarange("inner_%d" % i, 4 - i):
                    for k in range(n_superfluous_top + n_superfluous_bottom):
                        z_baseline = pyro.module("z_baseline_%d_%d" % (i, k),
                                                 pt_superfluous_baselines[3 * k + i], tags="baseline")
                        baseline_value = z_baseline(mu_latent.detach())
                        mean_i = pyro.param("mean_%d_%d" % (i, k),
                                            Variable(0.5 * torch.ones(4 - i), requires_grad=True))
                        z_i_k = pyro.sample("z_%d_%d" % (i, k),
                                            fakes.NonreparameterizedNormal(mean_i, 1),
                                            infer=dict(baseline=dict(baseline_value=baseline_value)))
                        assert z_i_k.shape == (4 - i,)

        def per_param_callable(module_name, param_name, tags):
            if 'baseline' in tags:
                return {"lr": 0.010, "betas": (0.95, 0.999)}
            else:
                return {"lr": 0.0012, "betas": (0.95, 0.999)}

        adam = optim.Adam(per_param_callable)
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

        for step in range(n_steps):
            svi.step()

            mu_error = param_abs_error("mu_q", self.analytic_mu_n)
            log_sig_error = param_abs_error("log_sig_q", self.analytic_log_sig_n)

            if n_superfluous_top > 0 or n_superfluous_bottom > 0:
                superfluous_errors = []
                for k in range(n_superfluous_top + n_superfluous_bottom):
                    mean_0_error = torch.sum(torch.pow(pyro.param("mean_0_%d" % k), 2.0))
                    mean_1_error = torch.sum(torch.pow(pyro.param("mean_1_%d" % k), 2.0))
                    mean_2_error = torch.sum(torch.pow(pyro.param("mean_2_%d" % k), 2.0))
                    superfluous_error = torch.max(torch.max(mean_0_error, mean_1_error), mean_2_error)
                    superfluous_errors.append(superfluous_error.detach().cpu().numpy())

            if step % 500 == 0:
                logger.debug("mu error, log(sigma) error:  %.4f, %.4f" % (mu_error, log_sig_error))
                if n_superfluous_top > 0 or n_superfluous_bottom > 0:
                    logger.debug("superfluous error: %.4f" % np.max(superfluous_errors))

        assert_equal(0.0, mu_error, prec=0.04)
        assert_equal(0.0, log_sig_error, prec=0.05)
        if n_superfluous_top > 0 or n_superfluous_bottom > 0:
            assert_equal(0.0, np.max(superfluous_errors), prec=0.04)
