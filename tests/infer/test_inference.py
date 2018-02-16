from __future__ import absolute_import, division, print_function

from unittest import TestCase

import pytest
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.distributions.testing import fakes
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedGamma
from pyro.distributions import TransformedDistribution
from pyro.infer.svi import SVI
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal
from tests.distributions.test_transformed_distribution import AffineExp


def param_mse(name, target):
    return torch.sum(torch.pow(target - pyro.param(name), 2.0)).data.cpu().numpy()[0]


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).data.cpu().numpy()[0]


class NormalNormalTests(TestCase):

    def setUp(self):
        # normal-normal; known covariance
        self.lam0 = Variable(torch.Tensor([0.1, 0.1]))   # precision of prior
        self.mu0 = Variable(torch.Tensor([0.0, 0.5]))   # prior mean
        # known precision of observation noise
        self.lam = Variable(torch.Tensor([6.0, 4.0]))
        self.data = Variable(torch.Tensor([[-0.1, 0.3],
                                           [0.00, 0.4],
                                           [0.20, 0.5],
                                           [0.10, 0.7]]))
        self.n_data = Variable(torch.Tensor([len(self.data)]))
        self.data_sum = self.data.sum(0)
        self.analytic_lam_n = self.lam0 + self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.data_sum * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)
        self.batch_size = 4

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 5000)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 15000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.clear_param_store()

        def model():
            mu_latent = pyro.sample("mu_latent",
                                    dist.Normal(self.mu0, torch.pow(self.lam0, -0.5)))
            pyro.sample("obs",
                        dist.Normal(mu_latent, torch.pow(self.lam, -0.5)),
                        obs=self.data)
            return mu_latent

        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.134 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.14 * torch.ones(2),
                                   requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
            pyro.sample("mu_latent", Normal(mu_q, sig_q))

        adam = optim.Adam({"lr": .001})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(n_steps):
            svi.step()

            mu_error = param_mse("mu_q", self.analytic_mu_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)

        assert_equal(0.0, mu_error, prec=0.05)
        assert_equal(0.0, log_sig_error, prec=0.05)


@pytest.mark.skip("Reinstate once poisson is migrated to PyTorch - https://github.com/uber/pyro/issues/699")
class TestFixedModelGuide(TestCase):
    def setUp(self):
        self.data = Variable(torch.Tensor([2.0]))
        self.alpha_q_log_0 = 0.17 * torch.ones(1)
        self.beta_q_log_0 = 0.19 * torch.ones(1)
        self.alpha_p_log_0 = 0.11 * torch.ones(1)
        self.beta_p_log_0 = 0.13 * torch.ones(1)

    def do_test_fixedness(self, fixed_tags):
        pyro.clear_param_store()

        def model():
            alpha_p_log = pyro.param(
                "alpha_p_log", Variable(
                    self.alpha_p_log_0.clone(), requires_grad=True), tags="model")
            beta_p_log = pyro.param(
                "beta_p_log", Variable(
                    self.beta_p_log_0.clone(), requires_grad=True), tags="model")
            alpha_p, beta_p = torch.exp(alpha_p_log), torch.exp(beta_p_log)
            lambda_latent = pyro.sample("lambda_latent", dist.Gamma(alpha_p, beta_p))
            pyro.sample("obs", dist.Poisson(lambda_latent), obs=self.data)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log", Variable(
                    self.alpha_q_log_0.clone(), requires_grad=True), tags="guide")
            beta_q_log = pyro.param(
                "beta_q_log", Variable(
                    self.beta_q_log_0.clone(), requires_grad=True), tags="guide")
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", dist.Gamma(alpha_q, beta_q))

        def per_param_args(module_name, param_name, tags):
            if tags in fixed_tags:
                    return {'lr': 0.0}
            else:
                return {'lr': 0.0}

        adam = optim.Adam(per_param_args)
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for _ in range(3):
            svi.step()

        model_unchanged = (torch.equal(pyro.param("alpha_p_log").data, self.alpha_p_log_0)) and\
                          (torch.equal(pyro.param("beta_p_log").data, self.beta_p_log_0))
        guide_unchanged = (torch.equal(pyro.param("alpha_q_log").data, self.alpha_q_log_0)) and\
                          (torch.equal(pyro.param("beta_q_log").data, self.beta_q_log_0))
        model_changed = not model_unchanged
        guide_changed = not guide_unchanged
        error = ('model' in fixed_tags and model_changed) or ('guide' in fixed_tags and guide_changed)
        return (not error)

    def test_model_fixed(self):
        assert self.do_test_fixedness(fixed_tags=["model"])

    def test_guide_fixed(self):
        assert self.do_test_fixedness(fixed_tags=["guide"])

    def test_guide_and_model_both_fixed(self):
        assert self.do_test_fixedness(fixed_tags=["model", "guide"])

    def test_guide_and_model_free(self):
        assert self.do_test_fixedness(fixed_tags=["bogus_tag"])


@pytest.mark.stage("integration", "integration_batch_2")
@pytest.mark.skip("Reinstate once poisson is migrated to PyTorch - https://github.com/uber/pyro/issues/699")
class PoissonGammaTests(TestCase):
    def setUp(self):
        # poisson-gamma model
        # gamma prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        # gamma prior hyperparameter
        self.beta0 = Variable(torch.Tensor([1.0]))
        self.data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
        self.n_data = len(self.data)
        data_sum = self.data.sum(0)
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 + \
            Variable(torch.Tensor([self.n_data]))  # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 10000)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 25000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.clear_param_store()
        Gamma = dist.Gamma if reparameterized else fakes.NonreparameterizedGamma

        def model():
            lambda_latent = pyro.sample("lambda_latent", Gamma(self.alpha0, self.beta0))
            pyro.sample("obs", dist.Poisson(lambda_latent), obs=self.data)
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
            pyro.sample("lambda_latent", Gamma(alpha_q, beta_q))

        adam = optim.Adam({"lr": .0002, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(n_steps):
            svi.step()

        alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
        beta_error = param_abs_error("beta_q_log", self.log_beta_n)
        assert_equal(0.0, alpha_error, prec=0.08)
        assert_equal(0.0, beta_error, prec=0.08)


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

    def test_elbo_reparameterized(self):
        self.do_elbo_test(dist.Gamma, 5000)

    def test_elbo_rsvi(self):
        self.do_elbo_test(ShapeAugmentedGamma, 5000)
        self.do_elbo_test(ShapeAugmentedGamma, 5000, True)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(fakes.NonreparameterizedGamma, 10000)

    def do_elbo_test(self, gamma_dist, n_steps, trace_graph=False):
        pyro.clear_param_store()

        def model():
            lambda_latent = pyro.sample("lambda_latent", gamma_dist(self.alpha0, self.beta0))
            pyro.sample("obs0", dist.Exponential(lambda_latent), obs=self.data[0])
            pyro.sample("obs1", dist.Exponential(lambda_latent), obs=self.data[1])
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log",
                Variable(self.log_alpha_n.data + 0.17, requires_grad=True))
            beta_q_log = pyro.param(
                "beta_q_log",
                Variable(self.log_beta_n.data - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", gamma_dist(alpha_q, beta_q))

        adam = optim.Adam({"lr": .0003, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=trace_graph)

        for k in range(n_steps):
            svi.step()

        alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
        beta_error = param_abs_error("beta_q_log", self.log_beta_n)
        assert_equal(0.0, alpha_error, prec=0.08)
        assert_equal(0.0, beta_error, prec=0.08)


class BernoulliBetaTests(TestCase):
    def setUp(self):
        # bernoulli-beta model
        # beta prior hyperparameter
        self.alpha0 = Variable(torch.Tensor([1.0]))
        self.beta0 = Variable(torch.Tensor([1.0]))  # beta prior hyperparameter
        self.data = Variable(torch.Tensor([[0.0], [1.0], [1.0], [1.0]]))
        self.n_data = len(self.data)
        self.batch_size = None
        data_sum = self.data.sum(0)
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 - data_sum + \
            Variable(torch.Tensor([self.n_data]))
        # posterior beta
        self.log_alpha_n = torch.log(self.alpha_n)
        self.log_beta_n = torch.log(self.beta_n)

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 10000)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 10000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.clear_param_store()
        Beta = dist.Beta if reparameterized else fakes.NonreparameterizedBeta

        def model():
            p_latent = pyro.sample("p_latent", Beta(self.alpha0, self.beta0))
            pyro.observe("obs", dist.Bernoulli(p_latent), obs=self.data)
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     Variable(self.log_alpha_n.data + 0.17, requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    Variable(self.log_beta_n.data - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("p_latent", Beta(alpha_q, beta_q))

        adam = optim.Adam({"lr": .001, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(n_steps):
            svi.step()

        alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
        beta_error = param_abs_error("beta_q_log", self.log_beta_n)
        assert_equal(0.0, alpha_error, prec=0.08)
        assert_equal(0.0, beta_error, prec=0.08)


class LogNormalNormalGuide(nn.Module):
    def __init__(self, mu_q_log_init, tau_q_log_init):
        super(LogNormalNormalGuide, self).__init__()
        self.mu_q_log = Parameter(mu_q_log_init)
        self.tau_q_log = Parameter(tau_q_log_init)


@pytest.mark.stage("integration", "integration_batch_2")
class LogNormalNormalTests(TestCase):
    def setUp(self):
        # lognormal-normal model
        # putting some of the parameters inside of a torch module to
        # make sure that that functionality is ok (XXX: do this somewhere else in the future)
        self.mu0 = Variable(torch.Tensor([1.0]))  # normal prior hyperparameter
        # normal prior hyperparameter
        self.tau0 = Variable(torch.Tensor([1.0]))
        # known precision for observation likelihood
        self.tau = Variable(torch.Tensor([2.5]))
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

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 15000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.clear_param_store()
        pt_guide = LogNormalNormalGuide(self.log_mu_n.data + 0.17,
                                        self.log_tau_n.data - 0.143)

        def model():
            mu_latent = pyro.sample("mu_latent",
                                    dist.Normal(self.mu0, torch.pow(self.tau0, -0.5)))
            sigma = torch.pow(self.tau, -0.5)
            pyro.observe("obs0", dist.LogNormal(mu_latent, sigma), obs=self.data[0])
            pyro.observe("obs1", dist.LogNormal(mu_latent, sigma), obs=self.data[1])
            return mu_latent

        def guide():
            pyro.module("mymodule", pt_guide)
            mu_q, tau_q = torch.exp(pt_guide.mu_q_log), torch.exp(pt_guide.tau_q_log)
            sigma = torch.pow(tau_q, -0.5)
            Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
            pyro.sample("mu_latent", Normal(mu_q, sigma))

        adam = optim.Adam({"lr": .0005, "betas": (0.96, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(n_steps):
            svi.step()

        mu_error = param_abs_error("mymodule$$$mu_q_log", self.log_mu_n)
        tau_error = param_abs_error("mymodule$$$tau_q_log", self.log_tau_n)
        assert_equal(0.0, mu_error, prec=0.07)
        assert_equal(0.0, tau_error, prec=0.07)

    def test_elbo_with_transformed_distribution(self):
        pyro.clear_param_store()

        def model():
            zero = Variable(torch.zeros(1))
            one = Variable(torch.ones(1))
            mu_latent = pyro.sample("mu_latent",
                                    dist.Normal(self.mu0, torch.pow(self.tau0, -0.5)))
            bijector = AffineExp(torch.pow(self.tau, -0.5), mu_latent)
            x_dist = TransformedDistribution(dist.Normal(zero, one), bijector)
            pyro.observe("obs0", x_dist, self.data[0])
            pyro.observe("obs1", x_dist, self.data[1])
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
            pyro.sample("mu_latent", dist.Normal(mu_q, torch.pow(tau_q, -0.5)))

        adam = optim.Adam({"lr": .0005, "betas": (0.96, 0.999)})
        svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)

        for k in range(12001):
            svi.step()

        mu_error = param_abs_error("mu_q_log", self.log_mu_n)
        tau_error = param_abs_error("tau_q_log", self.log_tau_n)
        assert_equal(0.0, mu_error, prec=0.05)
        assert_equal(0.0, tau_error, prec=0.05)


class SafetyTests(TestCase):

    def setUp(self):
        # normal-normal; known covariance
        def model_dup():
            pyro.param("mu_q", Variable(torch.ones(1), requires_grad=True))
            pyro.sample("mu_q", dist.Normal(ng_zeros(1), ng_ones(1)))

        def model_obs_dup():
            pyro.sample("mu_q", dist.Normal(ng_zeros(1), ng_ones(1)))
            pyro.sample("mu_q", dist.Normal(ng_zeros(1), ng_ones(1)), obs=ng_zeros(1))

        def model():
            pyro.sample("mu_q", dist.Normal(ng_zeros(1), ng_ones(1)))

        def guide():
            p = pyro.param("p", Variable(torch.ones(1), requires_grad=True))
            pyro.sample("mu_q", dist.Normal(ng_zeros(1), p))
            pyro.sample("mu_q_2", dist.Normal(ng_zeros(1), p))

        self.duplicate_model = model_dup
        self.duplicate_obs = model_obs_dup
        self.model = model
        self.guide = guide

    def test_duplicate_names(self):
        pyro.clear_param_store()

        adam = optim.Adam({"lr": .001})
        svi = SVI(self.duplicate_model, self.guide, adam, loss="ELBO", trace_graph=False)

        with pytest.raises(RuntimeError):
            svi.step()

    def test_extra_samples(self):
        pyro.clear_param_store()

        adam = optim.Adam({"lr": .001})
        svi = SVI(self.model, self.guide, adam, loss="ELBO", trace_graph=False)

        with pytest.warns(Warning):
            svi.step()

    def test_duplicate_obs_name(self):
        pyro.clear_param_store()

        adam = optim.Adam({"lr": .001})
        svi = SVI(self.duplicate_obs, self.guide, adam, loss="ELBO", trace_graph=False)

        with pytest.raises(RuntimeError):
            svi.step()
