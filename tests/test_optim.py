import torch
import torch.optim
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
from tests.common import TestCase

from pyro.infer.kl_qp import KL_QP


class OptimTests(TestCase):

    def setUp(self):
        # normal-normal; known covariance
        self.lam0 = Variable(torch.Tensor([0.1]))  # precision of prior
        self.mu0 = Variable(torch.Tensor([0.5]))  # prior mean
        # known precision of observation noise
        self.lam = Variable(torch.Tensor([6.0]))
        self.data = Variable(torch.Tensor([1.0]))  # a single observation

    def test_per_param_optim(self):
        self.do_test_per_param_optim("mu_q", "log_sig_q")
        self.do_test_per_param_optim("log_sig_q", "mu_q")

    # make sure lr=0 gets propagated correctly to parameters of our choice
    def do_test_per_param_optim(self, fixed_param, free_param):
        pyro._param_store._clear_cache()

        def model():
            prior_dist = DiagNormal(self.mu0, torch.pow(self.lam0, -0.5))
            mu_latent = pyro.sample("mu_latent", prior_dist)
            x_dist = DiagNormal(mu_latent, torch.pow(self.lam, -0.5))
            pyro.observe("obs", x_dist, self.data)
            return mu_latent

        def guide():
            mu_q = pyro.param(
                "mu_q",
                Variable(
                    torch.zeros(1),
                    requires_grad=True))
            log_sig_q = pyro.param(
                "log_sig_q", Variable(
                    torch.zeros(1), requires_grad=True))
            sig_q = torch.exp(log_sig_q)
            pyro.sample("mu_latent", DiagNormal(mu_q, sig_q))

        def optim_params(param_name, param):
            if param_name == fixed_param:
                return {'lr': 0.00}
            elif param_name == free_param:
                return {'lr': 0.01}

        kl_optim = KL_QP(
            model, guide, pyro.optim(
                torch.optim.Adam, optim_params))
        for k in range(3):
            kl_optim.step()

        free_param_unchanged = torch.equal(
            pyro.param(free_param).data, torch.zeros(1))
        fixed_param_unchanged = torch.equal(
            pyro.param(fixed_param).data, torch.zeros(1))
        passed_test = fixed_param_unchanged and not free_param_unchanged
        self.assertTrue(passed_test)
