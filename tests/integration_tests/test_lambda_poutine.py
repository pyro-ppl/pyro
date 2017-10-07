from __future__ import print_function

import pytest
import torch
import torch.optim
from torch.autograd import Variable
import pyro
import pyro.distributions as dist
from pyro.optim import Optimize
from pyro.util import ng_zeros
from tests.common import TestCase
import numpy as np

pytestmark = pytest.mark.stage("integration", "integration_batch_2")


class NormalNormalTests(TestCase):

    def setUp(self):
        torch.manual_seed(0)
        # normal-normal; known covariance
        self.lam0 = Variable(torch.Tensor([0.1, 0.1]))   # precision of prior
        self.mu0 = Variable(torch.Tensor([0.0, 0.5]))   # prior mean
        # known precision of observation noise
        self.lam = Variable(torch.Tensor([6.0, 4.0]))
        self.n_outer = 3
        self.n_inner = 3
        self.n_data = Variable(torch.Tensor([self.n_outer * self.n_inner]))
        self.data = []
        self.sum_data = ng_zeros(2)
        for _out in range(self.n_outer):
            data_in = []
            for _in in range(self.n_inner):
                data_in.append(Variable(torch.Tensor([-0.1, 0.3]) + torch.randn(2) / torch.sqrt(self.lam.data)))
                self.sum_data += data_in[-1]
            self.data.append(data_in)
        self.analytic_lam_n = self.lam0 + self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.sum_data * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)
        self.verbose = False

    # this tests rao-blackwellization in elbo for nested list map_datas
    def test_nested_list_map_data_in_elbo(self, n_steps=10000):
        pyro.get_param_store().clear()

        def model():
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.lam0, -0.5),
                                    reparameterized=False)

            def obs_outer(i, x):
                pyro.map_data("map_obs_inner_%d" % i, x, lambda _i, _x:
                              obs_inner(i, _i, _x), batch_size=3)

            def obs_inner(i, _i, _x):
                pyro.observe("obs_%d_%d" % (i, _i), dist.diagnormal, _x, mu_latent,
                             torch.pow(self.lam, -0.5))

            pyro.map_data("map_obs_outer", self.data, lambda i, x:
                          obs_outer(i, x), batch_size=3)

            return mu_latent

        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.184 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.19 * torch.ones(2),
                                   requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            mu_latent = pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q,
                                    reparameterized=False, use_avg_decaying_baseline=True)

            def obs_outer(i, x):
                pyro.map_data("map_obs_inner_%d" % i, x, lambda _i, _x:
                              None, batch_size=3)

            pyro.map_data("map_obs_outer", self.data, lambda i, x:
                          obs_outer(i, x), batch_size=3)

            return mu_latent

        guide_tracegraph = pyro.poutine.tracegraph(guide)()
        guide_trace = guide_tracegraph.get_trace()
        model_tracegraph = pyro.poutine.tracegraph(pyro.poutine.replay(model, guide_trace))()
        self.assertEqual(len(model_tracegraph.get_graph().edges()), 9)
        self.assertEqual(len(model_tracegraph.get_graph().nodes()), 10)
        self.assertEqual(len(guide_tracegraph.get_graph().edges()), 0)
        self.assertEqual(len(guide_tracegraph.get_graph().nodes()), 1)

        optim = Optimize(model, guide,
                         torch.optim.Adam, {"lr": .0008, "betas": (0.96, 0.999)},
                         loss="ELBO", trace_graph=True)

        for k in range(n_steps):
            optim.step()

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
            if k % 500 == 0 and self.verbose:
                print("mu error, log(sigma) error:  %.4f, %.4f" % (mu_error.data.numpy()[0],
                      log_sig_error.data.numpy()[0]))

        self.assertEqual(0.0, mu_error.data.cpu().numpy()[0], prec=0.04)
        self.assertEqual(0.0, log_sig_error.data.cpu().numpy()[0], prec=0.04)

    # this tests rao-blackwellization and baselines for a vectorized map_data
    # inside of a list map_data with superfluous random variables to complexify the
    # graph structure and introduce additional baselines
    def test_vectorized_map_data_in_elbo_with_superfluous_rvs(self):
        self._test_vectorized_map_data_in_elbo(n_superfluous_top=2, n_superfluous_bottom=2, n_steps=6000)

    def _test_vectorized_map_data_in_elbo(self, n_superfluous_top, n_superfluous_bottom, n_steps):
        pyro.get_param_store().clear()
        self.data_tensor = Variable(torch.zeros(9, 2))
        for _out in range(self.n_outer):
            for _in in range(self.n_inner):
                self.data_tensor[3 * _out + _in, :] = self.data[_out][_in]

        def model():
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.lam0, -0.5),
                                    reparameterized=False)

            def obs_inner(i, _i, _x):
                for k in range(n_superfluous_top):
                    pyro.sample("z_%d_%d" % (i, k), dist.diagnormal, ng_zeros(4 - i, 1),
                                ng_ones(4 - i, 1), reparameterized=False)
                pyro.observe("obs_%d" % i, dist.diagnormal, _x, mu_latent, torch.pow(self.lam, -0.5))
                for k in range(n_superfluous_top, n_superfluous_top + n_superfluous_bottom):
                    pyro.sample("z_%d_%d" % (i, k), dist.diagnormal, ng_zeros(4 - i, 1),
                                ng_ones(4 - i, 1), reparameterized=False)

            def obs_outer(i, x):
                pyro.map_data("map_obs_inner_%d" % i, x, lambda _i, _x:
                              obs_inner(i, _i, _x), batch_size=4 - i)

            pyro.map_data("map_obs_outer", [self.data_tensor[0:4, :], self.data_tensor[4:7, :],
                                            self.data_tensor[7:9, :]],
                          lambda i, x: obs_outer(i, x), batch_size=3)

            return mu_latent

        pt_mu_baseline = torch.nn.Linear(1, 1)
        pt_superfluous_baselines = []
        for k in range(n_superfluous_top + n_superfluous_bottom):
            pt_superfluous_baselines.extend([torch.nn.Linear(2, 4), torch.nn.Linear(2, 3),
                                             torch.nn.Linear(2, 2)])

        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.184 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.19 * torch.ones(2), requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            trivial_baseline = pyro.module("mu_baseline", pt_mu_baseline)
            baseline_value = trivial_baseline(ng_ones(1))
            baseline_params = trivial_baseline.parameters()
            mu_latent = pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q, baseline_value=baseline_value,
                                    baseline_params=baseline_params, reparameterized=False)

            def obs_inner(i, _i, _x):
                for k in range(n_superfluous_top + n_superfluous_bottom):
                    z_baseline = pyro.module("z_baseline_%d_%d" % (i, k),
                                             pt_superfluous_baselines[3 * k + i])
                    baseline_value = z_baseline(mu_latent.detach())
                    baseline_params = z_baseline.parameters()
                    mean_i = pyro.param("mean_%d_%d" % (i, k),
                                        Variable(0.5 * torch.ones(4 - i, 1), requires_grad=True))
                    pyro.sample("z_%d_%d" % (i, k), dist.diagnormal, mean_i, ng_ones(4 - i, 1),
                                baseline_value=baseline_value, baseline_params=baseline_params,
                                reparameterized=False)

            def obs_outer(i, x):
                pyro.map_data("map_obs_inner_%d" % i, x, lambda _i, _x:
                              obs_inner(i, _i, _x), batch_size=4 - i)

            pyro.map_data("map_obs_outer", [self.data_tensor[0:4, :], self.data_tensor[4:7, :],
                                            self.data_tensor[7:9, :]],
                          lambda i, x: obs_outer(i, x), batch_size=3)

            return mu_latent

        kl_optim = TraceGraph_KL_QP(model, guide, pyro.optim(
                                    torch.optim.Adam,
                                    {"lr": .0012, "betas": (0.96, 0.999)}))

        for step in range(n_steps):
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
            if n_superfluous_top > 0 or n_superfluous_bottom > 0:
                superfluous_errors = []
                for k in range(n_superfluous_top + n_superfluous_bottom):
                    mean_0_error = torch.sum(torch.pow(pyro.param("mean_0_%d" % k), 2.0))
                    mean_1_error = torch.sum(torch.pow(pyro.param("mean_1_%d" % k), 2.0))
                    mean_2_error = torch.sum(torch.pow(pyro.param("mean_2_%d" % k), 2.0))
                    superfluous_error = torch.max(torch.max(mean_0_error, mean_1_error), mean_2_error)
                    superfluous_errors.append(superfluous_error.data.numpy()[0])

            if step % 500 == 0 and self.verbose:
                print("mu error, log(sigma) error:  %.4f, %.4f" % (mu_error.data.numpy()[0],
                      log_sig_error.data.numpy()[0]))
                if n_superfluous_top > 0 or n_superfluous_bottom > 0:
                    print("superfluous error: %.4f" % np.max(superfluous_errors))

        self.assertEqual(0.0, mu_error.data.cpu().numpy()[0], prec=0.04)
        self.assertEqual(0.0, log_sig_error.data.cpu().numpy()[0], prec=0.04)
        if n_superfluous_top > 0 or n_superfluous_bottom > 0:
            self.assertEqual(0.0, np.max(superfluous_errors), prec=0.04)
