from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.infer
from pyro.distributions import Normal
from tests.common import TestCase
from pyro.infer.abstract_infer import Marginal
from pyro.infer.mcmc.mh import MH, NormalProposal
from pyro.infer.mcmc.mcmc import MCMC
from pyro.util import ng_ones, ng_zeros


class NormalNormalSamplingTestCase(TestCase):

    def setUp(self):

        pyro.clear_param_store()

        def model():
            mu = pyro.sample("mu", Normal(Variable(torch.zeros(1)),
                                          Variable(torch.ones(1))))
            xd = Normal(mu, Variable(torch.ones(1)), batch_size=50)
            pyro.observe("xs", xd, self.data)
            return mu

        # data
        self.data = Variable(torch.zeros(50, 1))
        self.mu_mean = Variable(torch.zeros(1))
        self.mu_stddev = torch.sqrt(Variable(torch.ones(1)) / 51.0)

        # model and guide
        self.model = model


class MHTest(NormalNormalSamplingTestCase):

    @pytest.mark.init(rng_seed=0)
    def test_single_site_mh(self):

        # basic mcmc run with normal normal
        mcmc_run = MCMC(MH(self.model, NormalProposal(mu=ng_zeros(1), sigma=ng_ones(1), tune_frequency=100)),
                        num_samples=1000, warmup_steps=150)

        marginal = Marginal(mcmc_run)
        posterior_samples = [marginal() for i in range(1000)]

        posterior_mean = torch.mean(torch.cat(posterior_samples))
        posterior_stddev = torch.std(torch.cat(posterior_samples), 0)

        self.assertEqual(0, torch.norm(posterior_mean - self.mu_mean).data[0],
                         prec=0.03)
        self.assertEqual(0, torch.norm(posterior_stddev - self.mu_stddev).data[0],
                         prec=0.1)


if __name__ == "__main__":
    # generic test
    nnt = MHTest()
    nnt.setUp()
    nnt.test_single_site_mh()
