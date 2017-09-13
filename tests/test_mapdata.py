import torch
import torch.optim
from torch.autograd import Variable
from torch import nn as nn
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.transformed_distribution import AffineExp, TransformedDistribution
from pyro.infer.importance import Importance
from tests.common import TestCase

from pyro.infer.kl_qp import KL_QP


class NormalNormalTests(TestCase):

    def setUp(self):
                    # normal-normal; known covariance
        self.lam0 = Variable(torch.Tensor([0.1, 0.1]))   # precision of prior
        self.mu0 = Variable(torch.Tensor([0.0, 0.5]))   # prior mean
        # known precision of observation noise
        self.lam = Variable(torch.Tensor([6.0, 4.0]))
        self.data = []
        self.sum_data = Variable(torch.zeros(2))
        for i in range(8):
            self.data.append(Variable(torch.Tensor([0.3, 0.1]) + 1.2 * torch.randn(2) / torch.sqrt(self.lam0.data)))
            self.sum_data.data.add_(self.data[-1].data)

        self.n_data = Variable(torch.Tensor([len(self.data)]))
        self.analytic_lam_n = self.lam0 + \
            self.n_data.expand_as(self.lam) * self.lam
        self.analytic_log_sig_n = -0.5 * torch.log(self.analytic_lam_n)
        self.analytic_mu_n = self.sum_data * (self.lam / self.analytic_lam_n) +\
            self.mu0 * (self.lam0 / self.analytic_lam_n)
        self.verbose = True
        self.batch_sizes = [3, 7, 8, 0]

    def test_elbo_tensor(self):
        for batch_size in self.batch_sizes:
            self.do_elbo_test(True, 10000, batch_size, map_type="tensor")

    def test_elbo_none(self):
        self.do_elbo_test(True, 10000, 0, map_type=None)

    def test_elbo_list(self):
        for batch_size in self.batch_sizes:
            self.do_elbo_test(True, 10000, batch_size, map_type="list")

    def do_elbo_test(self, reparameterized, n_steps, batch_size, map_type):
        if self.verbose:
            print("DOING ELBO TEST [repa = {}, bs = {}, map_type = {}]".format(
                reparameterized, batch_size, map_type))
        pyro.get_param_store().clear()

        def model():
            mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                    self.mu0, torch.pow(self.lam0, -0.5))
            if map_type == "list":
                pyro.map_data("aaa", self.data, lambda i,
                              x: pyro.observe(
                                  "obs_%d" % i, dist.diagnormal,
                                  x, mu_latent, torch.pow(self.lam, -0.5)), batch_size=batch_size)
                pyro.map_data("bbb", self.data, lambda i,
                              x: pyro.sample(
                                  "z_sample_%d" % i, dist.diagnormal,
                                  x, torch.pow(self.lam, -0.5)), batch_size=batch_size)
            elif map_type == "tensor":
                tdata = torch.cat([xi.view(1, -1) for xi in self.data], 0)
                pyro.map_data("aaa", tdata,
                              # XXX get batch size args to dist right
                              lambda i, x: pyro.observe("obs", dist.diagnormal, x, mu_latent,
                                                        torch.pow(self.lam, -0.5)),
                              batch_size=batch_size)
                pyro.map_data("bbb", tdata,
                              # XXX get batch_size args to dist right
                              lambda i, x: pyro.sample("z_sample", dist.diagnormal,
                                                       x, torch.pow(self.lam, -0.5)),
                              batch_size=batch_size)
            else:
                for i, x in enumerate(self.data):
                    pyro.observe('obs_%d' % i, dist.diagnormal, x, mu_latent, torch.pow(self.lam, -0.5))
            return mu_latent

        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.434 * torch.randn(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                self.analytic_log_sig_n.data - 0.39 * torch.randn(2),
                requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q)
            if map_type == "list" or map_type is None:
                pyro.map_data("aaa", self.data, lambda i, x: None, batch_size=batch_size)
                pyro.map_data("bbb", self.data,
                              lambda i, x: pyro.sample(
                                  "z_sample_%d" % i, dist.diagnormal,
                                  x, torch.pow(self.lam, -0.5)), batch_size=batch_size)
            elif map_type == "tensor":
                tdata = torch.cat([xi.view(1, -1) for xi in self.data], 0)
                # dummy map_data to do subsampling for observe
                pyro.map_data("aaa", tdata, lambda i, x: None, batch_size=batch_size)
                pyro.map_data("bbb", tdata,
                              # XXX get batch_size args to dist right
                              lambda i, x: pyro.sample("z_sample", dist.diagnormal,
                                                       x, torch.pow(self.lam, -0.5)),
                              batch_size=batch_size)

            else:
                pass

        kl_optim = KL_QP(
            model, guide, pyro.optim(
                torch.optim.Adam, {"lr": 0.008, "betas": (0.95, 0.999)}))
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

            if self.verbose and k % 500 == 0:
                print("errors", mu_error.data.numpy()[0], log_sig_error.data.numpy()[0])

        self.assertEqual(0.0, mu_error.data.cpu().numpy()[0], prec=0.05)
        self.assertEqual(0.0, log_sig_error.data.cpu().numpy()[0], prec=0.05)
