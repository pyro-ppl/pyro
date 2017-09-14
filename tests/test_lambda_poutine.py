from __future__ import print_function
import torch
import torch.optim
from torch.autograd import Variable
from torch import nn as nn
from torch.nn import Parameter
import numpy as np

import pyro
import pyro.distributions as dist
from tests.common import TestCase
import pyro.poutine as poutine

from pyro.infer.tracegraph_kl_qp import TraceGraph_KL_QP
from pyro.util import ng_ones, ng_zeros, ones, zeros
from pyro.distributions.transformed_distribution import AffineExp, TransformedDistribution


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
        self.data.append(Variable(torch.Tensor([-0.2, 0.0])))
        self.data.append(Variable(torch.Tensor([0.30, 0.4])))
        self.data.append(Variable(torch.Tensor([0.10, -0.3])))
        self.data.append(Variable(torch.Tensor([0.17, 0.1])))
        self.data.append(Variable(torch.Tensor([-0.03, 0.03])))
        self.data.append(Variable(torch.Tensor([0.15, 0.22])))
        self.data.append(Variable(torch.Tensor([0.17, -0.09])))
        self.data.append(Variable(torch.Tensor([0.07, 0.11])))
        self.n_data = Variable(torch.Tensor([len(self.data)]))
        self.sum_data = self.data[0] + self.data[1] + self.data[2] + self.data[3] + \
            self.data[4] + self.data[5] + self.data[6] + self.data[7] + \
            self.data[8] + self.data[9] + self.data[10] + self.data[11]
        self.data = [(self.data[0], self.data[1], self.data[2]),
                     (self.data[3], self.data[4], self.data[5]),
                     (self.data[6], self.data[7], self.data[8]),
                     (self.data[9], self.data[10], self.data[11])]
        self.analytic_lam_n = self.lam0 + \
            self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.sum_data * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)
        self.verbose = True

    def test_elbo_reparameterized(self):
    #    self.do_elbo_test(True, 1000)
        self.do_elbo_test(False, 20000)

    #def test_elbo_nonreparameterized(self):
    #    self.do_elbo_test(False, 5000)

    def do_elbo_test(self, reparameterized, n_steps):
        if self.verbose:
            print(" - - - - - DO NORMALNORMAL ELBO TEST  [reparameterized = %s] - - - - - " % reparameterized)
        pyro.get_param_store().clear()

        def model():
            #print("entering model")
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.lam0, -0.5),
                                    reparameterized=reparameterized)
            #pyro.map_data("map_obs_outer", self.data, lambda i, x:
            #              pyro.map_data("map_obs_inner_%d" % i, [(x, i)], lambda _i, _x:
            #                  pyro.observe("obs_%d" % _x[1], dist.diagnormal, _x[0], mu_latent,
            #                               torch.pow(self.lam, -0.5)), batch_size=1), batch_size=2)

            def obs(i, x1, x2, x3):
                pyro.observe("obs_%d_1" % i, dist.diagnormal, x1, mu_latent, torch.pow(self.lam, -0.5))
                pyro.observe("obs_%d_2" % i, dist.diagnormal, x2, mu_latent, torch.pow(self.lam, -0.5))
                pyro.observe("obs_%d_3" % i, dist.diagnormal, x3, mu_latent, torch.pow(self.lam, -0.5))

            pyro.map_data("map_obs", self.data, lambda i, x: obs(i, x[0], x[1], x[2]), batch_size=2)
            #print("exiting model")
            return mu_latent

        def guide():
            #print("\n**********\nentering guide")
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.334 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.29 * torch.ones(2),
                                   requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            mu_latent = pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q,
                                    reparameterized=reparameterized,
                                    avg_decaying_baseline=True, baseline_beta=0.95)
            #pyro.map_data("map_obs_outer", self.data, lambda i, x:
            #        pyro.map_data("map_obs_inner_%d" % i, [(x, i)], lambda _i, _x: None, batch_size=1),
            #                      batch_size=2)
            #pyro.map_data("map_obs", self.data, lambda i, x: None, batch_size=2)
            pyro.map_data("map_obs", self.data, lambda i, x: None, batch_size=2)
            #print("exiting guide")
            return mu_latent

        guide_tracegraph = poutine.tracegraph(guide)()
        guide_tracegraph.save_visualization('guide')
        guide_trace = guide_tracegraph.get_trace()
        model_tracegraph = poutine.tracegraph(poutine.replay(model, guide_trace))()
        model_tracegraph.save_visualization('model')
        model_trace = model_tracegraph.get_trace()

        kl_optim = TraceGraph_KL_QP(model, guide, pyro.optim(
                                    torch.optim.Adam,
                                    {"lr": .001, "betas": (0.95, 0.999)}))

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

        self.assertEqual(0.0, mu_error.data.cpu().numpy()[0], prec=0.03)
        self.assertEqual(0.0, log_sig_error.data.cpu().numpy()[0], prec=0.03)

