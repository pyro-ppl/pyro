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
import pyro.infer
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

        def model():
            mu = pyro.sample("mu", DiagNormal(Variable(torch.zeros(1)),
                                              Variable(torch.ones(1))))
            xd = DiagNormal(mu, Variable(torch.ones(1)))
            xs = pyro.map_data("aa", self.data,
                               lambda i, x_i: pyro.observe("x{}".format(i), xd, x_i))
            return xs

        def guide():
            pyro.map_data("aa", self.data, lambda i, x_i: None)
            return pyro.sample("mu", DiagNormal(Variable(torch.zeros(1)),
                                                Variable(torch.ones(1))))

        # data
        self.data = [Variable(torch.zeros(1)) for i in range(50)]
        self.mu_mean = Variable(torch.zeros(1))
        self.mu_stddev = torch.sqrt(Variable(torch.ones(1)) / 51.0)

        # model and guide
        self.model = model
        self.guide = guide


class MHTest(NormalNormalSamplingTestCase):

    def test_mh_guide(self):
        posterior = pyro.infer.MH(self.model, guide=self.guide,
                                  samples=1000, lag=1, burn=0)
        posterior_samples = [posterior()[0][0]["mu"]["value"] for i in range(500)]
        posterior_mean = torch.mean(torch.cat(posterior_samples))
        posterior_stddev = torch.sqrt(torch.mean(torch.cat(posterior_samples) ** 2))
        self.assertEqual(0, torch.norm(posterior_mean - self.mu_mean).data[0],
                         prec=0.01)
        self.assertEqual(0, torch.norm(posterior_stddev - self.mu_stddev).data[0],
                         prec=0.01)

    # def test_mh_single_site(self):
    #     posterior = pyro.infer.mh.SingleSiteMH(self.model, samples=1000)
    #     tr = posterior()
    #     self.assertTrue(tr is not None)


class ImportanceTest(NormalNormalSamplingTestCase):

    def test_importance_guide(self):
        posterior = pyro.infer.Importance(self.model, guide=self.guide, samples=2000)
        posterior_samples = [posterior()[0][0]["mu"]["value"] for i in range(500)]
        posterior_mean = torch.mean(torch.cat(posterior_samples))
        posterior_stddev = torch.sqrt(torch.mean(torch.cat(posterior_samples) ** 2))
        self.assertEqual(0, torch.norm(posterior_mean - self.mu_mean).data[0],
                         prec=0.01)
        self.assertEqual(0, torch.norm(posterior_stddev - self.mu_stddev).data[0],
                         prec=0.01)


class SearchTest(HMMSamplingTestCase):

    def test_complete(self):
        pdb.set_trace()
        posterior = pyro.infer.Search(self.model)
        traces, log_weights = posterior._traces()
