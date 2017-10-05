from __future__ import print_function

import pytest
import torch
import torch.optim
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer.tracegraph_kl_qp import TraceGraph_KL_QP
from pyro.util import ng_zeros
from tests.common import TestCase

pytestmark = pytest.mark.integration_test


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

    def test_nested_map_data_in_elbo(self, n_steps=10000):
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
                                    reparameterized=False)

            def obs_outer(i, x):
                pyro.map_data("map_obs_inner_%d" % i, x, lambda _i, _x:
                              None, batch_size=3)

            pyro.map_data("map_obs_outer", self.data, lambda i, x:
                          obs_outer(i, x), batch_size=3)

            return mu_latent

        guide_tracegraph = pyro.poutine.tracegraph(guide).get_trace()
        guide_trace = guide_tracegraph.get_trace()
        model_tracegraph = pyro.poutine.tracegraph(pyro.poutine.replay(model, guide_trace)).get_trace()
        self.assertEqual(len(model_tracegraph.get_graph().edges()), 9)
        self.assertEqual(len(model_tracegraph.get_graph().nodes()), 10)
        self.assertEqual(len(guide_tracegraph.get_graph().edges()), 0)
        self.assertEqual(len(guide_tracegraph.get_graph().nodes()), 1)

        kl_optim = TraceGraph_KL_QP(model, guide, pyro.optim(
                                    torch.optim.Adam,
                                    {"lr": .0008, "betas": (0.96, 0.999)}))
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
            if k % 500 == 0 and self.verbose:
                print("mu error, log(sigma) error:  %.4f, %.4f" % (mu_error.data.numpy()[0],
                      log_sig_error.data.numpy()[0]))

        self.assertEqual(0.0, mu_error.data.cpu().numpy()[0], prec=0.04)
        self.assertEqual(0.0, log_sig_error.data.cpu().numpy()[0], prec=0.04)
