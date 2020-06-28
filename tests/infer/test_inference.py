# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from unittest import TestCase

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.contrib.gp.kernels as kernels
import pyro.distributions as dist
import pyro.optim as optim
from pyro import poutine
from pyro.distributions.testing import fakes
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedGamma
from pyro.infer import (SVI, EnergyDistance, JitTrace_ELBO, JitTraceEnum_ELBO, JitTraceGraph_ELBO, RenyiELBO,
                        ReweightedWakeSleep, Trace_ELBO, Trace_MMD, TraceEnum_ELBO, TraceGraph_ELBO,
                        TraceMeanField_ELBO, TraceTailAdaptive_ELBO)
from pyro.infer.autoguide import AutoDelta
from pyro.infer.reparam import LatentStableReparam
from pyro.infer.util import torch_item
from tests.common import assert_close, assert_equal, xfail_if_not_implemented, xfail_param

logger = logging.getLogger(__name__)


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
        self.sample_batch_size = 2

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 5000, Trace_ELBO())

    def test_elbo_analytic_kl(self):
        self.do_elbo_test(True, 3000, TraceMeanField_ELBO())

    def test_elbo_tail_adaptive(self):
        self.do_elbo_test(True, 3000, TraceTailAdaptive_ELBO(num_particles=10, vectorize_particles=True))

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 15000, Trace_ELBO())

    def test_renyi_reparameterized(self):
        self.do_elbo_test(True, 2500, RenyiELBO(num_particles=3, vectorize_particles=False))

    def test_renyi_nonreparameterized(self):
        self.do_elbo_test(False, 7500, RenyiELBO(num_particles=3, vectorize_particles=True))

    def test_rws_reparameterized(self):
        self.do_elbo_test(True, 2500, ReweightedWakeSleep(num_particles=3))

    def test_rws_nonreparameterized(self):
        self.do_elbo_test(False, 7500, ReweightedWakeSleep(num_particles=3))

    def test_mmd_vectorized(self):
        z_size = self.loc0.shape[0]
        self.do_fit_prior_test(
            True, 1000, Trace_MMD(
                kernel=kernels.RBF(
                    z_size,
                    lengthscale=torch.sqrt(torch.tensor(z_size, dtype=torch.float))
                ), vectorize_particles=True, num_particles=100
            )
        )

    def test_mmd_nonvectorized(self):
        z_size = self.loc0.shape[0]
        self.do_fit_prior_test(
            True, 100, Trace_MMD(
                kernel=kernels.RBF(
                    z_size,
                    lengthscale=torch.sqrt(torch.tensor(z_size, dtype=torch.float))
                ), vectorize_particles=False, num_particles=100
            ), lr=0.0146
        )

    def do_elbo_test(self, reparameterized, n_steps, loss):
        pyro.clear_param_store()

        def model():
            loc_latent = pyro.sample("loc_latent",
                                     dist.Normal(self.loc0, torch.pow(self.lam0, -0.5))
                                     .to_event(1))
            with pyro.plate('data', self.batch_size):
                pyro.sample("obs",
                            dist.Normal(loc_latent, torch.pow(self.lam, -0.5)).to_event(1),
                            obs=self.data)
            return loc_latent

        def guide():
            loc_q = pyro.param("loc_q", self.analytic_loc_n.detach() + 0.134)
            log_sig_q = pyro.param("log_sig_q", self.analytic_log_sig_n.data.detach() - 0.14)
            sig_q = torch.exp(log_sig_q)
            Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
            pyro.sample("loc_latent", Normal(loc_q, sig_q).to_event(1))

        adam = optim.Adam({"lr": .001})
        svi = SVI(model, guide, adam, loss=loss)

        for k in range(n_steps):
            svi.step()

            loc_error = param_mse("loc_q", self.analytic_loc_n)
            log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)

        assert_equal(0.0, loc_error, prec=0.05)
        assert_equal(0.0, log_sig_error, prec=0.05)

    def do_fit_prior_test(self, reparameterized, n_steps, loss, debug=False, lr=0.001):
        pyro.clear_param_store()

        def model():
            with pyro.plate('samples', self.sample_batch_size):
                pyro.sample(
                    "loc_latent", dist.Normal(
                        torch.stack([self.loc0]*self.sample_batch_size, dim=0),
                        torch.stack([torch.pow(self.lam0, -0.5)]*self.sample_batch_size, dim=0)
                    ).to_event(1)
                )

        def guide():
            loc_q = pyro.param("loc_q", self.loc0.detach() + 0.134)
            log_sig_q = pyro.param("log_sig_q", -0.5*torch.log(self.lam0).data.detach() - 0.14)
            sig_q = torch.exp(log_sig_q)
            Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
            with pyro.plate('samples', self.sample_batch_size):
                pyro.sample(
                    "loc_latent", Normal(
                        torch.stack([loc_q]*self.sample_batch_size, dim=0),
                        torch.stack([sig_q]*self.sample_batch_size, dim=0)
                    ).to_event(1)
                )

        adam = optim.Adam({"lr": lr})
        svi = SVI(model, guide, adam, loss=loss)

        alpha = 0.99
        for k in range(n_steps):
            svi.step()
            if debug:
                loc_error = param_mse("loc_q", self.loc0)
                log_sig_error = param_mse("log_sig_q", -0.5*torch.log(self.lam0))
                with torch.no_grad():
                    if k == 0:
                        avg_loglikelihood, avg_penalty = loss._differentiable_loss_parts(model, guide)
                        avg_loglikelihood = torch_item(avg_loglikelihood)
                        avg_penalty = torch_item(avg_penalty)
                    loglikelihood, penalty = loss._differentiable_loss_parts(model, guide)
                    avg_loglikelihood = alpha * avg_loglikelihood + (1-alpha) * torch_item(loglikelihood)
                    avg_penalty = alpha * avg_penalty + (1-alpha) * torch_item(penalty)
                if k % 100 == 0:
                    print(loc_error, log_sig_error)
                    print(avg_loglikelihood, avg_penalty)
                    print()

        loc_error = param_mse("loc_q", self.loc0)
        log_sig_error = param_mse("log_sig_q", -0.5 * torch.log(self.lam0))
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
                "alpha_p_log", self.alpha_p_log_0.clone())
            beta_p_log = pyro.param(
                "beta_p_log", self.beta_p_log_0.clone())
            alpha_p, beta_p = torch.exp(alpha_p_log), torch.exp(beta_p_log)
            lambda_latent = pyro.sample("lambda_latent", dist.Gamma(alpha_p, beta_p))
            pyro.sample("obs", dist.Poisson(lambda_latent), obs=self.data)
            return lambda_latent

        def guide():
            alpha_q_log = pyro.param(
                "alpha_q_log", self.alpha_q_log_0.clone())
            beta_q_log = pyro.param(
                "beta_q_log", self.beta_q_log_0.clone())
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
        self.sample_batch_size = 2

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 10000, Trace_ELBO())

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 25000, Trace_ELBO())

    def test_renyi_reparameterized(self):
        self.do_elbo_test(True, 5000, RenyiELBO(num_particles=2))

    def test_renyi_nonreparameterized(self):
        self.do_elbo_test(False, 12500, RenyiELBO(alpha=0.2, num_particles=2))

    def test_rws_reparameterized(self):
        self.do_elbo_test(True, 5000, ReweightedWakeSleep(num_particles=2))

    def test_rws_nonreparameterized(self):
        self.do_elbo_test(False, 12500, ReweightedWakeSleep(num_particles=2))

    def test_mmd_vectorized(self):
        z_size = 1
        self.do_fit_prior_test(
            True, 500, Trace_MMD(
                kernel=kernels.RBF(
                    z_size,
                    lengthscale=torch.sqrt(torch.tensor(z_size, dtype=torch.float))
                ), vectorize_particles=True, num_particles=100
            ), debug=True, lr=0.09
        )

    def do_elbo_test(self, reparameterized, n_steps, loss):
        pyro.clear_param_store()
        Gamma = dist.Gamma if reparameterized else fakes.NonreparameterizedGamma

        def model():
            lambda_latent = pyro.sample("lambda_latent", Gamma(self.alpha0, self.beta0))
            with pyro.plate("data", self.n_data):
                pyro.sample("obs", dist.Poisson(lambda_latent), obs=self.data)
            return lambda_latent

        def guide():
            alpha_q = pyro.param("alpha_q", self.alpha_n.detach() + math.exp(0.17),
                                 constraint=constraints.positive)
            beta_q = pyro.param("beta_q", self.beta_n.detach() / math.exp(0.143),
                                constraint=constraints.positive)
            pyro.sample("lambda_latent", Gamma(alpha_q, beta_q))

        adam = optim.Adam({"lr": .0002, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss)

        for k in range(n_steps):
            svi.step()

        assert_equal(pyro.param("alpha_q"), self.alpha_n, prec=0.2, msg='{} vs {}'.format(
            pyro.param("alpha_q").detach().cpu().numpy(), self.alpha_n.detach().cpu().numpy()))
        assert_equal(pyro.param("beta_q"), self.beta_n, prec=0.15, msg='{} vs {}'.format(
            pyro.param("beta_q").detach().cpu().numpy(), self.beta_n.detach().cpu().numpy()))

    def do_fit_prior_test(self, reparameterized, n_steps, loss, debug=False, lr=0.0002):
        pyro.clear_param_store()
        Gamma = dist.Gamma if reparameterized else fakes.NonreparameterizedGamma

        def model():
            with pyro.plate('samples', self.sample_batch_size):
                pyro.sample(
                    "lambda_latent", Gamma(
                        torch.stack([torch.stack([self.alpha0])]*self.sample_batch_size),
                        torch.stack([torch.stack([self.beta0])]*self.sample_batch_size)
                    ).to_event(1)
                )

        def guide():
            alpha_q = pyro.param("alpha_q", self.alpha0.detach() + math.exp(0.17),
                                 constraint=constraints.positive)
            beta_q = pyro.param("beta_q", self.beta0.detach() / math.exp(0.143),
                                constraint=constraints.positive)
            with pyro.plate('samples', self.sample_batch_size):
                pyro.sample(
                    "lambda_latent", Gamma(
                        torch.stack([torch.stack([alpha_q])]*self.sample_batch_size),
                        torch.stack([torch.stack([beta_q])]*self.sample_batch_size)
                    ).to_event(1)
                )

        adam = optim.Adam({"lr": lr, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss)

        alpha = 0.99
        for k in range(n_steps):
            svi.step()
            if debug:
                alpha_error = param_mse("alpha_q", self.alpha0)
                beta_error = param_mse("beta_q", self.beta0)
                with torch.no_grad():
                    if k == 0:
                        avg_loglikelihood, avg_penalty = loss._differentiable_loss_parts(model, guide, (), {})
                        avg_loglikelihood = torch_item(avg_loglikelihood)
                        avg_penalty = torch_item(avg_penalty)
                    loglikelihood, penalty = loss._differentiable_loss_parts(model, guide, (), {})
                    avg_loglikelihood = alpha * avg_loglikelihood + (1-alpha) * torch_item(loglikelihood)
                    avg_penalty = alpha * avg_penalty + (1-alpha) * torch_item(penalty)
                if k % 100 == 0:
                    print(alpha_error, beta_error)
                    print(avg_loglikelihood, avg_penalty)
                    print()

        assert_equal(pyro.param("alpha_q"), self.alpha0, prec=0.2, msg='{} vs {}'.format(
            pyro.param("alpha_q").detach().cpu().numpy(), self.alpha0.detach().cpu().numpy()))
        assert_equal(pyro.param("beta_q"), self.beta0, prec=0.15, msg='{} vs {}'.format(
            pyro.param("beta_q").detach().cpu().numpy(), self.beta0.detach().cpu().numpy()))


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.parametrize('elbo_impl', [
    xfail_param(JitTrace_ELBO, reason="incorrect gradients", run=False),
    xfail_param(JitTraceGraph_ELBO, reason="incorrect gradients", run=False),
    xfail_param(JitTraceEnum_ELBO, reason="incorrect gradients", run=False),
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceEnum_ELBO,
    RenyiELBO,
    ReweightedWakeSleep
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
    prec = 0.2 if gamma_dist.has_rsample else 0.25

    def model(alpha0, beta0, alpha_n, beta_n):
        lambda_latent = pyro.sample("lambda_latent", gamma_dist(alpha0, beta0))
        with pyro.plate("data", n_data):
            pyro.sample("obs", dist.Exponential(lambda_latent), obs=data)
        return lambda_latent

    def guide(alpha0, beta0, alpha_n, beta_n):
        alpha_q = pyro.param("alpha_q", alpha_n * math.exp(0.17), constraint=constraints.positive)
        beta_q = pyro.param("beta_q", beta_n / math.exp(0.143), constraint=constraints.positive)
        pyro.sample("lambda_latent", gamma_dist(alpha_q, beta_q))

    adam = optim.Adam({"lr": .0003, "betas": (0.97, 0.999)})
    if elbo_impl is RenyiELBO:
        elbo = elbo_impl(alpha=0.2, num_particles=3, max_plate_nesting=1, strict_enumeration_warning=False)
    elif elbo_impl is ReweightedWakeSleep:
        if gamma_dist is ShapeAugmentedGamma:
            pytest.xfail(reason="ShapeAugmentedGamma not suported for ReweightedWakeSleep")
        else:
            elbo = elbo_impl(num_particles=3, max_plate_nesting=1, strict_enumeration_warning=False)
    else:
        elbo = elbo_impl(max_plate_nesting=1, strict_enumeration_warning=False)
    svi = SVI(model, guide, adam, loss=elbo)

    with xfail_if_not_implemented():
        for k in range(n_steps):
            svi.step(alpha0, beta0, alpha_n, beta_n)

    assert_equal(pyro.param("alpha_q"), alpha_n, prec=prec, msg='{} vs {}'.format(
        pyro.param("alpha_q").detach().cpu().numpy(), alpha_n.detach().cpu().numpy()))
    assert_equal(pyro.param("beta_q"), beta_n, prec=prec, msg='{} vs {}'.format(
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
        self.sample_batch_size = 2

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 10000, Trace_ELBO())

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 10000, Trace_ELBO())

    # this is used to detect bugs related to https://github.com/pytorch/pytorch/issues/9521
    def test_elbo_reparameterized_vectorized(self):
        self.do_elbo_test(True, 5000, Trace_ELBO(num_particles=2, vectorize_particles=True,
                                                 max_plate_nesting=1))

    # this is used to detect bugs related to https://github.com/pytorch/pytorch/issues/9521
    def test_elbo_nonreparameterized_vectorized(self):
        self.do_elbo_test(False, 5000, Trace_ELBO(num_particles=2, vectorize_particles=True,
                                                  max_plate_nesting=1))

    def test_renyi_reparameterized(self):
        self.do_elbo_test(True, 5000, RenyiELBO(num_particles=2))

    def test_renyi_nonreparameterized(self):
        self.do_elbo_test(False, 5000, RenyiELBO(alpha=0.2, num_particles=2))

    def test_renyi_reparameterized_vectorized(self):
        self.do_elbo_test(True, 5000, RenyiELBO(num_particles=2, vectorize_particles=True,
                                                max_plate_nesting=1))

    def test_renyi_nonreparameterized_vectorized(self):
        self.do_elbo_test(False, 5000, RenyiELBO(alpha=0.2, num_particles=2, vectorize_particles=True,
                                                 max_plate_nesting=1))

    def test_rws_reparameterized(self):
        self.do_elbo_test(True, 5000, ReweightedWakeSleep(num_particles=2))

    def test_rws_nonreparameterized(self):
        self.do_elbo_test(False, 5000, ReweightedWakeSleep(num_particles=2))

    def test_rws_reparameterized_vectorized(self):
        self.do_elbo_test(True, 5000, ReweightedWakeSleep(num_particles=2, vectorize_particles=True,
                                                          max_plate_nesting=1))

    def test_rws_nonreparameterized_vectorized(self):
        self.do_elbo_test(False, 5000, ReweightedWakeSleep(num_particles=2, vectorize_particles=True,
                                                           max_plate_nesting=1))

    def test_mmd_vectorized(self):
        z_size = 1
        self.do_fit_prior_test(
            True, 2500, Trace_MMD(
                kernel=kernels.RBF(
                    z_size,
                    lengthscale=torch.sqrt(torch.tensor(z_size, dtype=torch.float))
                ), vectorize_particles=True, num_particles=100
            )
        )

    def do_elbo_test(self, reparameterized, n_steps, loss):
        pyro.clear_param_store()
        Beta = dist.Beta if reparameterized else fakes.NonreparameterizedBeta

        def model():
            p_latent = pyro.sample("p_latent", Beta(self.alpha0, self.beta0))
            with pyro.plate("data", self.batch_size):
                pyro.sample("obs", dist.Bernoulli(p_latent), obs=self.data)
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     self.log_alpha_n + 0.17)
            beta_q_log = pyro.param("beta_q_log",
                                    self.log_beta_n - 0.143)
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            pyro.sample("p_latent", Beta(alpha_q, beta_q))

        adam = optim.Adam({"lr": .001, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss=loss)

        for k in range(n_steps):
            svi.step()

        alpha_error = param_abs_error("alpha_q_log", self.log_alpha_n)
        beta_error = param_abs_error("beta_q_log", self.log_beta_n)
        assert_equal(0.0, alpha_error, prec=0.08)
        assert_equal(0.0, beta_error, prec=0.08)

    def do_fit_prior_test(self, reparameterized, n_steps, loss, debug=False):
        pyro.clear_param_store()
        Beta = dist.Beta if reparameterized else fakes.NonreparameterizedBeta

        def model():
            with pyro.plate('samples', self.sample_batch_size):
                pyro.sample(
                    "p_latent", Beta(
                        torch.stack([torch.stack([self.alpha0])]*self.sample_batch_size),
                        torch.stack([torch.stack([self.beta0])]*self.sample_batch_size)
                    ).to_event(1)
                )

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     torch.log(self.alpha0) + 0.17)
            beta_q_log = pyro.param("beta_q_log",
                                    torch.log(self.beta0) - 0.143)
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            with pyro.plate('samples', self.sample_batch_size):
                pyro.sample(
                    "p_latent", Beta(
                        torch.stack([torch.stack([alpha_q])]*self.sample_batch_size),
                        torch.stack([torch.stack([beta_q])]*self.sample_batch_size)
                    ).to_event(1)
                )

        adam = optim.Adam({"lr": .001, "betas": (0.97, 0.999)})
        svi = SVI(model, guide, adam, loss=loss)

        alpha = 0.99
        for k in range(n_steps):
            svi.step()
            if debug:
                alpha_error = param_abs_error("alpha_q_log", torch.log(self.alpha0))
                beta_error = param_abs_error("beta_q_log", torch.log(self.beta0))
                with torch.no_grad():
                    if k == 0:
                        avg_loglikelihood, avg_penalty = loss._differentiable_loss_parts(model, guide)
                        avg_loglikelihood = torch_item(avg_loglikelihood)
                        avg_penalty = torch_item(avg_penalty)
                    loglikelihood, penalty = loss._differentiable_loss_parts(model, guide)
                    avg_loglikelihood = alpha * avg_loglikelihood + (1-alpha) * torch_item(loglikelihood)
                    avg_penalty = alpha * avg_penalty + (1-alpha) * torch_item(penalty)
                if k % 100 == 0:
                    print(alpha_error, beta_error)
                    print(avg_loglikelihood, avg_penalty)
                    print()

        alpha_error = param_abs_error("alpha_q_log", torch.log(self.alpha0))
        beta_error = param_abs_error("beta_q_log", torch.log(self.beta0))
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


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.parametrize("prior_scale", [0, 1e-4])
def test_energy_distance_univariate(prior_scale):

    def model(data):
        loc = pyro.sample("loc", dist.Normal(0, 100))
        scale = pyro.sample("scale", dist.LogNormal(0, 1))
        with pyro.plate("data", len(data)):
            pyro.sample("obs", dist.Normal(loc, scale), obs=data)

    def guide(data):
        loc_loc = pyro.param("loc_loc", torch.tensor(0.))
        loc_scale = pyro.param("loc_scale", torch.tensor(1.),
                               constraint=constraints.positive)
        log_scale_loc = pyro.param("log_scale_loc", torch.tensor(0.))
        log_scale_scale = pyro.param("log_scale_scale", torch.tensor(1.),
                                     constraint=constraints.positive)
        pyro.sample("loc", dist.Normal(loc_loc, loc_scale))
        pyro.sample("scale", dist.LogNormal(log_scale_loc, log_scale_scale))

    data = 10.0 + torch.randn(8)
    adam = optim.Adam({"lr": 0.1})
    loss_fn = EnergyDistance(num_particles=32, prior_scale=prior_scale)
    svi = SVI(model, guide, adam, loss_fn)
    for step in range(2001):
        loss = svi.step(data)
        if step % 20 == 0:
            logger.info("step {} loss = {:0.4g}, loc = {:0.4g}, scale = {:0.4g}"
                        .format(step, loss, pyro.param("loc_loc").item(),
                                pyro.param("log_scale_loc").exp().item()))

    expected_loc = data.mean()
    expected_scale = data.std()
    actual_loc = pyro.param("loc_loc").detach()
    actual_scale = pyro.param("log_scale_loc").exp().detach()
    assert_close(actual_loc, expected_loc, atol=0.05)
    assert_close(actual_scale, expected_scale, rtol=0.1 if prior_scale else 0.05)


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.parametrize("prior_scale", [0, 1])
def test_energy_distance_multivariate(prior_scale):

    def model(data):
        loc = torch.zeros(2)
        cov = pyro.sample("cov", dist.Normal(0, 100).expand([2, 2]).to_event(2))
        with pyro.plate("data", len(data)):
            pyro.sample("obs", dist.MultivariateNormal(loc, cov), obs=data)

    def guide(data):
        scale_tril = pyro.param("scale_tril", torch.eye(2),
                                constraint=constraints.lower_cholesky)
        pyro.sample("cov", dist.Delta(scale_tril @ scale_tril.t(), event_dim=2))

    cov = torch.tensor([[1, 0.8], [0.8, 1]])
    data = dist.MultivariateNormal(torch.zeros(2), cov).sample([10])
    loss_fn = EnergyDistance(num_particles=32, prior_scale=prior_scale)
    svi = SVI(model, guide, optim.Adam({"lr": 0.1}), loss_fn)
    for step in range(2001):
        loss = svi.step(data)
        if step % 20 == 0:
            logger.info("step {} loss = {:0.4g}".format(step, loss))

    delta = data - data.mean(0)
    expected_cov = (delta.t() @ delta) / len(data)
    scale_tril = pyro.param("scale_tril").detach()
    actual_cov = scale_tril @ scale_tril.t()
    assert_close(actual_cov, expected_cov, atol=0.2)


@pytest.mark.stage("integration", "integration_batch_1")
def test_reparam_stable():
    data = dist.Poisson(torch.randn(8).exp()).sample()

    @poutine.reparam(config={"dz": LatentStableReparam(), "y": LatentStableReparam()})
    def model():
        stability = pyro.sample("stability", dist.Uniform(1., 2.))
        trans_skew = pyro.sample("trans_skew", dist.Uniform(-1., 1.))
        obs_skew = pyro.sample("obs_skew", dist.Uniform(-1., 1.))
        scale = pyro.sample("scale", dist.Gamma(3, 1))

        # We use separate plates because the .cumsum() op breaks independence.
        with pyro.plate("time1", len(data)):
            dz = pyro.sample("dz", dist.Stable(stability, trans_skew))
        z = dz.cumsum(-1)
        with pyro.plate("time2", len(data)):
            y = pyro.sample("y", dist.Stable(stability, obs_skew, scale, z))
            pyro.sample("x", dist.Poisson(y.abs()), obs=data)

    guide = AutoDelta(model)
    svi = SVI(model, guide, optim.Adam({"lr": 0.01}), Trace_ELBO())
    for step in range(100):
        loss = svi.step()
        if step % 20 == 0:
            logger.info("step {} loss = {:0.4g}".format(step, loss))


@pytest.mark.stage("integration", "integration_batch_1")
def test_sequential_plating_sum():
    """Example from https://github.com/pyro-ppl/pyro/issues/2361"""

    def model(data):
        x = pyro.sample('x', dist.Bernoulli(torch.tensor(0.5)))
        for i in pyro.plate('data_plate', len(data)):
            pyro.sample('data_{:d}'.format(i),
                        dist.Normal(x, scale=torch.tensor(0.1)),
                        obs=data[i])

    def guide(data):
        p = pyro.param('p', torch.tensor(0.5))
        pyro.sample('x', pyro.distributions.Bernoulli(p))

    data = torch.cat([torch.randn([5]), 1. + torch.randn([5])])
    adam = optim.Adam({"lr": 0.01})
    loss_fn = RenyiELBO(alpha=0, num_particles=30, vectorize_particles=True)
    svi = SVI(model, guide, adam, loss_fn)

    for step in range(1):
        loss = svi.step(data)
        if step % 20 == 0:
            logger.info("step {} loss = {:0.4g}".format(step, loss))
