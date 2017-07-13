import numpy as np
import torch
import pdb
import sys
from torch.autograd import Variable
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

import pyro
from pyro.distributions import DiagNormal, Bernoulli
import pyro.poutine as poutine
from pyro.util import memoize

from tests.common import TestCase


class HMMSamplingTestCase(TestCase):

    def setUp(self):

        # simple Gaussian-mixture HMM
        def model():
            ps = pyro.param("ps", Variable(torch.Tensor([[0.8], [0.3]])))
            mu = pyro.param("mu", Variable(torch.Tensor([[-0.1], [0.9]])))
            sigma = Variable(torch.ones(1))

            latents = [Variable(torch.ones(1))]
            observes = []
            for t in range(5):

                latents.append(
                    pyro.sample("latent_{}".format(str(t)),
                                Bernoulli(ps[latents[-1][0].long().data])))

                observes.append(
                    pyro.observe("observe_{}".format(str(t)),
                                 DiagNormal(mu[latents[-1][0].long().data], sigma),
                                 pyro.ones(1)))
            return latents

        self.model = model


class NormalNormalSamplingTestCase(TestCase):

    def setUp(self):

        pyro._param_store._clear_cache()

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

        def model():
            prior_dist = DiagNormal(self.mu0, torch.pow(self.lam0, -0.5))
            mu_latent = pyro.sample("mu_latent", prior_dist)
            x_dist = DiagNormal(mu_latent, torch.pow(self.lam, -0.5))
            # x = pyro.observe("obs", x_dist, self.data)
            pyro.map_data("aaa", self.data, lambda i,
                          x: pyro.observe("obs_%d" % i, x_dist, x), batch_size=1)
            return mu_latent

        def guide():
            mu_q = pyro.param("mu_q", Variable(self.analytic_mu_n.data + 0.134 * torch.ones(2),
                                               requires_grad=True))
            log_sig_q = pyro.param("log_sig_q", Variable(
                                   self.analytic_log_sig_n.data - 0.09 * torch.ones(2),
                                   requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            q_dist = DiagNormal(mu_q, sig_q)
            q_dist.reparametrized = reparametrized
            pyro.sample("mu_latent", q_dist)
            pyro.map_data("aaa", self.data, lambda i, x: None, batch_size=1)

        # model and guide
        self.model = model
        self.guide = guide


class MHTest(NormalNormalSamplingTestCase):
    pass
