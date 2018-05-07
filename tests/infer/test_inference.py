from __future__ import absolute_import, division, print_function

import math
from unittest import TestCase

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.distributions.testing import fakes
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedGamma
from pyro.infer import (SVI, JitTrace_ELBO, JitTraceEnum_ELBO, JitTraceGraph_ELBO, Trace_ELBO, TraceEnum_ELBO,
                        TraceGraph_ELBO)
from tests.common import assert_equal, xfail_param


def param_mse(name, target):
    return torch.sum(torch.pow(target - pyro.param(name), 2.0)).item()


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).item()


@pytest.mark.stage("integration", "integration_batch_1")
class NormalNormalTests(TestCase):

    def setUp(self):
        # normal-normal; known covariance
        self.lam0 = torch.tensor([0.1, 0.1])   # precision of prior
        self.loc0 = torch.tensor([0.0, 0.5])   # prior mean
        # known precision of observation noise
        self.lam = torch.tensor([6.0, 4.0])
        self.data = torch.tensor([[-0.1, 0.3],
                                  [0.00, 0.4],
                                  [0.20, 0.5],
                                  [0.10, 0.7]])
        self.n_data = torch.tensor([float(len(self.data))])
        self.data_sum = self.data.sum(0)
        self.analytic_lam_n = self.lam0 + self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_loc_n = self.data_sum * (self.lam / self.analytic_lam_n) +\
            self.loc0 * (self.lam0 / self.analytic_lam_n)
        self.batch_size = 4

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 5000)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 15000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.clear_param_store()

        def model():
            loc_latent = pyro.sample("loc_latent",
                                     dist.Normal(self.loc0, torch.pow(self.lam0, -0.5))
                                     .independent(1))
            with pyro.iarange('data', self.batch_size):
                pyro.sample("obs",
                            dist.Normal(loc_latent, torch.pow(self.lam, -0.5)).independent(1),
                            obs=self.data)
            return loc_latent

        def guide():
            loc_q = pyro.param("loc_q", torch.tensor(self.analytic_loc_n.data + 0.134 * torch.ones(2),
                                                     requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", torch.tensor(
                                   self.analytic_log_sig_n.data - 0.14 * torch.ones(2),
                                   requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
            pyro.sample("loc_latent", Normal(loc_q, sig_q).independent(1))

        adam = optim.Adam({"lr": .001})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        for k in range(n_steps):
            svi.step()

            loc_error = param_mse("loc_q", self.analytic_loc_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)

        assert_equal(0.0, loc_error, prec=0.05)
        assert_equal(0.0, log_sig_error, prec=0.05)


class TestFixedModelGuide(TestCase):
    def setUp(self):
        self.data = torch.tensor([2.0])
        self.alpha_q_log_0 = 0.17 * torch.ones(1)
        self.beta_q_log_0 = 0.19 * torch.ones(1)
        self.alpha_p_log_0 = 0.11 * torch.ones(1)
        self.beta_p_log_0 = 0.13 * torch.ones(1)

    def do_test_fixedness(self, fixed_parts):
        pyro.clear_param_store()

        def model():
            alpha_p_log = pyro.param(
                "alpha_p_log", torch.tensor(
                    self.alpha_p_log_0.clone()))
            beta_p_log = pyro.param(
                "beta_p_log", torch.tensor(
                    self.beta_p_log_0.clone()))
            alpha_p, beta_p = torch.exp(alpha_p_log), torch.exp(beta_p_log)
            lambda_latent = pyro.sample("lambda_latent", dist.Gamma(alpha_p, beta_p))
            pyro.sample("obs", dist.Poisson(lambda_latent), obs=self.data)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log", torch.tensor(
                    self.alpha_q_log_0.clone()))
            beta_q_log = pyro.param(
                "beta_q_log", torch.tensor(
                    self.beta_q_log_0.clone()))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("lambda_latent", dist.Gamma(alpha_q, beta_q))

        def per_param_args(module_name, param_name):
            if 'model' in fixed_parts and 'p_' in param_name:
                return {'lr': 0.0}
            if 'guide' in fixed_parts and 'q_' in param_name:
                return {'lr': 0.0}
            return {'lr': 0.01}

        adam = optim.Adam(per_param_args)
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        for _ in range(3):
            svi.step()

        model_unchanged = (torch.equal(pyro.param("alpha_p_log").data, self.alpha_p_log_0)) and\
                          (torch.equal(pyro.param("beta_p_log").data, self.beta_p_log_0))
        guide_unchanged = (torch.equal(pyro.param("alpha_q_log").data, self.alpha_q_log_0)) and\
                          (torch.equal(pyro.param("beta_q_log").data, self.beta_q_log_0))
        model_changed = not model_unchanged
        guide_changed = not guide_unchanged
        error = ('model' in fixed_parts and model_changed) or ('guide' in fixed_parts and guide_changed)
        return (not error)

    def test_model_fixed(self):
        assert self.do_test_fixedness(fixed_parts=["model"])

    def test_guide_fixed(self):
        assert self.do_test_fixedness(fixed_parts=["guide"])

    def test_guide_and_model_both_fixed(self):
        assert self.do_test_fixedness(fixed_parts=["model", "guide"])

    def test_guide_and_model_free(self):
        assert self.do_test_fixedness(fixed_parts=["bogus_tag"])


@pytest.mark.stage("integration", "integration_batch_2")
class PoissonGammaTests(TestCase):
    def setUp(self):
        # poisson-gamma model
        # gamma prior hyperparameter
        self.alpha0 = torch.tensor(1.0)
        # gamma prior hyperparameter
        self.beta0 = torch.tensor(1.0)
        self.data = torch.tensor([1.0, 2.0, 3.0])
        self.n_data = len(self.data)
        data_sum = self.data.sum(0)
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 + torch.tensor(float(self.n_data))  # posterior beta

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 10000)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 25000)

    def do_elbo_test(self, reparameterized, n_steps):
        pyro.clear_param_store()
        Gamma = dist.Gamma if reparameterized else fakes.NonreparameterizedGamma

        def model():
            lambda_latent = pyro.sample("lambda_latent", Gamma(self.alpha0, self.beta0))
            with pyro.iarange("data", self.n_data):
                pyro.sample("obs", dist.Poisson(lambda_latent), obs=self.data)
            return lambda_latent

        def guide():
            alpha_q = pyro.param("alpha_q", self.alpha_n.detach() + math.exp(0.17),
                                 constraint=constraints.positive)
            beta_q = pyro.param("beta_q", self.beta_n.detach() / math.exp(0.143),
                                constraint=constraints.positive)
            pyro.sample("lambda_latent", Gamma(alpha_q, beta_q))

        adam = optim.Adam({"lr": .0002, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        for k in range(n_steps):
            svi.step()

        assert_equal(pyro.param("alpha_q"), self.alpha_n, prec=0.2, msg='{} vs {}'.format(
            pyro.param("alpha_q").detach().cpu().numpy(), self.alpha_n.detach().cpu().numpy()))
        assert_equal(pyro.param("beta_q"), self.beta_n, prec=0.15, msg='{} vs {}'.format(
            pyro.param("beta_q").detach().cpu().numpy(), self.beta_n.detach().cpu().numpy()))


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.parametrize('elbo_impl', [
    xfail_param(JitTrace_ELBO, reason="incorrect gradients", run=False),
    xfail_param(JitTraceGraph_ELBO, reason="incorrect gradients", run=False),
    xfail_param(JitTraceEnum_ELBO, reason="incorrect gradients", run=False),
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceEnum_ELBO,
])
@pytest.mark.parametrize('gamma_dist,n_steps', [
    (dist.Gamma, 5000),
    (fakes.NonreparameterizedGamma, 10000),
    (ShapeAugmentedGamma, 5000),
], ids=['reparam', 'nonreparam', 'rsvi'])
def test_exponential_gamma(gamma_dist, n_steps, elbo_impl):
    pyro.clear_param_store()

    # gamma prior hyperparameter
    alpha0 = torch.tensor(1.0)
    # gamma prior hyperparameter
    beta0 = torch.tensor(1.0)
    n_data = 2
    data = torch.tensor([3.0, 2.0])  # two observations
    alpha_n = alpha0 + torch.tensor(float(n_data))  # posterior alpha
    beta_n = beta0 + torch.sum(data)  # posterior beta

    def model(alpha0, beta0, alpha_n, beta_n):
        lambda_latent = pyro.sample("lambda_latent", gamma_dist(alpha0, beta0))
        with pyro.iarange("data", n_data):
            pyro.sample("obs", dist.Exponential(lambda_latent), obs=data)
        return lambda_latent

    def guide(alpha0, beta0, alpha_n, beta_n):
        alpha_q = pyro.param("alpha_q", alpha_n * math.exp(0.17), constraint=constraints.positive)
        beta_q = pyro.param("beta_q", beta_n / math.exp(0.143), constraint=constraints.positive)
        pyro.sample("lambda_latent", gamma_dist(alpha_q, beta_q))

    adam = optim.Adam({"lr": .0003, "betas": (0.97, 0.999)})
    elbo = elbo_impl(strict_enumeration_warning=False)
    svi = SVI(model, guide, adam, loss=elbo, max_iarange_nesting=1)

    for k in range(n_steps):
        svi.step(alpha0, beta0, alpha_n, beta_n)

    assert_equal(pyro.param("alpha_q"), alpha_n, prec=0.15, msg='{} vs {}'.format(
        pyro.param("alpha_q").detach().cpu().numpy(), alpha_n.detach().cpu().numpy()))
    assert_equal(pyro.param("beta_q"), beta_n, prec=0.15, msg='{} vs {}'.format(
        pyro.param("beta_q").detach().cpu().numpy(), beta_n.detach().cpu().numpy()))


@pytest.mark.stage("integration", "integration_batch_2")
class BernoulliBetaTests(TestCase):
    def setUp(self):
        # bernoulli-beta model
        # beta prior hyperparameter
        self.alpha0 = torch.tensor(1.0)
        self.beta0 = torch.tensor(1.0)  # beta prior hyperparameter
        self.data = torch.tensor([0.0, 1.0, 1.0, 1.0])
        self.n_data = len(self.data)
        self.batch_size = 4
        data_sum = self.data.sum()
        self.alpha_n = self.alpha0 + data_sum  # posterior alpha
        self.beta_n = self.beta0 - data_sum + torch.tensor(float(self.n_data))
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
            with pyro.iarange("data", self.batch_size):
                pyro.sample("obs", dist.Bernoulli(p_latent), obs=self.data)
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     torch.tensor(self.log_alpha_n.data + 0.17, requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    torch.tensor(self.log_beta_n.data - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("p_latent", Beta(alpha_q, beta_q))

        adam = optim.Adam({"lr": .001, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        for k in range(n_steps):
            svi.step()

        alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
        beta_error = param_abs_error("beta_q_log", self.log_beta_n)
        assert_equal(0.0, alpha_error, prec=0.08)
        assert_equal(0.0, beta_error, prec=0.08)


class SafetyTests(TestCase):

    def setUp(self):
        # normal-normal; known covariance
        def model_dup():
            pyro.param("loc_q", torch.ones(1, requires_grad=True))
            pyro.sample("loc_q", dist.Normal(torch.zeros(1), torch.ones(1)))

        def model_obs_dup():
            pyro.sample("loc_q", dist.Normal(torch.zeros(1), torch.ones(1)))
            pyro.sample("loc_q", dist.Normal(torch.zeros(1), torch.ones(1)), obs=torch.zeros(1))

        def model():
            pyro.sample("loc_q", dist.Normal(torch.zeros(1), torch.ones(1)))

        def guide():
            p = pyro.param("p", torch.ones(1, requires_grad=True))
            pyro.sample("loc_q", dist.Normal(torch.zeros(1), p))
            pyro.sample("loc_q_2", dist.Normal(torch.zeros(1), p))

        self.duplicate_model = model_dup
        self.duplicate_obs = model_obs_dup
        self.model = model
        self.guide = guide

    def test_duplicate_names(self):
        pyro.clear_param_store()

        adam = optim.Adam({"lr": .001})
        svi = SVI(self.duplicate_model, self.guide, adam, loss=Trace_ELBO())

        with pytest.raises(RuntimeError):
            svi.step()

    def test_extra_samples(self):
        pyro.clear_param_store()

        adam = optim.Adam({"lr": .001})
        svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

        with pytest.warns(Warning):
            svi.step()

    def test_duplicate_obs_name(self):
        pyro.clear_param_store()

        adam = optim.Adam({"lr": .001})
        svi = SVI(self.duplicate_obs, self.guide, adam, loss=Trace_ELBO())

        with pytest.raises(RuntimeError):
            svi.step()
