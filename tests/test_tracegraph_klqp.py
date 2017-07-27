import torch
import torch.optim
from torch.autograd import Variable
from torch import nn as nn
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from tests.common import TestCase

from pyro.infer.tracegraph_kl_qp import TraceGraph_KL_QP


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

    def test_elbo_reparameterized(self):
        self.do_elbo_test(True, 1000)

    def test_elbo_nonreparameterized(self):
        self.do_elbo_test(False, 7000)

    def do_elbo_test(self, reparameterized, n_steps):
        print " - - - - - DO NORMALNORMAL ELBO TEST  [reparameterized = %s] - - - - - " % reparameterized
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

        kl_optim = TraceGraph_KL_QP(
            model, guide, pyro.optim(
                torch.optim.Adam,
                    #"lr": .002}))
                    {"lr": .001, "betas": (0.95, 0.999)}))
        for k in range(n_steps):
            kl_optim.step(step_number=k, reparameterized=reparameterized)

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
            if k%500==0:
                print "mu error, log(sigma) error:  %.4f, %.4f" % (mu_error.data.numpy()[0],
                      log_sig_error.data.numpy()[0])

        self.assertEqual(0.0, mu_error.data.cpu().numpy()[0], prec=0.03)
        self.assertEqual(0.0, log_sig_error.data.cpu().numpy()[0], prec=0.04)

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

    def test_elbo_nonreparameterized(self):
        print " - - - - - DO BERNOULLI-BETA ELBO TEST - - - - - "
        pyro.get_param_store().clear()

        def model():
            p_latent = pyro.sample("p_latent", dist.beta, self.alpha0, self.beta0)
	    for i, x in enumerate(self.data):
            	pyro.observe("obs_{}".format(i), dist.bernoulli, x, torch.pow(torch.pow(p_latent,2.0),0.5))
            return p_latent

        def guide():
            alpha_q_log = pyro.param("alpha_q_log",
                                     Variable(self.log_alpha_n.data + 0.17, requires_grad=True))
            beta_q_log = pyro.param("beta_q_log",
                                    Variable(self.log_beta_n.data - 0.143, requires_grad=True))
            alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
            p_latent = pyro.sample("p_latent", dist.beta, alpha_q, beta_q)
	    return p_latent

        kl_optim = TraceGraph_KL_QP(model, guide, pyro.optim(torch.optim.Adam,
                                                  {"lr": .001, "betas": (0.95, 0.999)}))
        for k in range(9001):
            kl_optim.step(step_number=k, reparameterized=False)
	    alpha_error = torch.abs(
		pyro.param("alpha_q_log") -
		self.log_alpha_n).data.cpu().numpy()[0]
	    beta_error = torch.abs(
		pyro.param("beta_q_log") -
		self.log_beta_n).data.cpu().numpy()[0]
	    if k%500==0:
		print "alpha, beta_error: ", alpha_error, beta_error

        self.assertEqual(0.0, alpha_error, prec=0.05)
        self.assertEqual(0.0, beta_error, prec=0.05)
