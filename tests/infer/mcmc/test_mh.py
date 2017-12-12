from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.infer
from pyro.distributions import Bernoulli, Normal
from tests.common import TestCase
from pyro.infer.abstract_infer import Marginal
from pyro.infer.mcmc.mh import MH, NormalProposal
from pyro.infer.mcmc.mcmc import MCMC
from pyro.util import ng_ones, ng_zeros


class HMMSamplingTestCase(TestCase):

    def setUp(self):

        # simple Gaussian-emission HMM
        def model():
            p_latent = pyro.param("p1", Variable(torch.Tensor([[0.7], [0.3]])))
            p_obs = pyro.param("p2", Variable(torch.Tensor([[0.9], [0.1]])))

            latents = [Variable(torch.ones(1, 1))]
            observes = []
            for t in range(self.model_steps):

                latents.append(
                    pyro.sample("latent_{}".format(str(t)),
                                Bernoulli(torch.index_select(p_latent, 0, latents[-1].view(-1).long()))))

                observes.append(
                    pyro.observe("observe_{}".format(str(t)),
                                 Bernoulli(torch.index_select(p_obs, 0, latents[-1].view(-1).long())),
                                 self.data[t]))
            return torch.sum(torch.cat(latents))

        self.model_steps = 3
        self.data = [pyro.ones(1, 1) for _ in range(self.model_steps)]
        self.model = model


class NormalNormalSamplingTestCase(TestCase):

    def setUp(self):

        pyro.clear_param_store()

        def model():
            mu = pyro.sample("mu", Normal(Variable(torch.zeros(1)),
                                          Variable(torch.ones(1))))
            xd = Normal(mu, Variable(torch.ones(1)), batch_size=50)
            pyro.observe("xs", xd, self.data)
            return mu

        def guide():
            return pyro.sample("mu", Normal(Variable(torch.zeros(1)),
                                            Variable(torch.ones(1))))

        # data
        self.data = Variable(torch.zeros(50, 1))
        self.mu_mean = Variable(torch.zeros(1))
        self.mu_stddev = torch.sqrt(Variable(torch.ones(1)) / 51.0)

        # model and guide
        self.model = model
        self.guide = guide


class SearchTest(HMMSamplingTestCase):

    def test_complete(self):
        posterior = pyro.infer.Search(self.model)

        true_latents = set()
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    true_latents.add((float(i1), float(i2), float(i3)))

        tr_latents = set()
        for tr, _ in posterior._traces():
            tr_latents.add(tuple([tr.nodes[name]["value"].view(-1).data[0]
                                  for name in tr.nodes.keys()
                                  if tr.nodes[name]["type"] == "sample" and
                                  not tr.nodes[name]["is_observed"]]))

        assert true_latents == tr_latents

    def test_marginal(self):
        posterior = pyro.infer.Search(self.model)
        marginal = pyro.infer.Marginal(posterior)
        d, values = marginal._dist_and_values()

        tr_rets = []
        for v in values:
            tr_rets.append(v.view(-1).data[0])

        assert len(tr_rets) == 4
        for i in range(4):
            assert i + 1 in tr_rets


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
        # print("Mean : {}, analytic: {}".format(posterior_mean, self.mu_mean))
        # print("std : {}, analytic: {}".format(posterior_stddev, self.mu_stddev))

        self.assertEqual(0, torch.norm(posterior_mean - self.mu_mean).data[0],
                         prec=0.03)
        self.assertEqual(0, torch.norm(posterior_stddev - self.mu_stddev).data[0],
                         prec=0.1)

    # def do_elbo_test(self, reparameterized, n_steps):
    #     pyro.clear_param_store()

    #     def model():
    #         mu_latent = pyro.sample("mu_latent", dist.normal,
    #                                 self.mu0, torch.pow(self.lam0, -0.5))

    #         map(lambda i, x: pyro.observe("obs_%d" % i, dist.normal,
    #                                       x, mu_latent, torch.pow(self.lam, -0.5)),
    #             enumerate(self.data))

    #         return mu_latent

    #     mh_params = {'mu': ng_zeros(1), 'sigma': ng_ones(1)*5, 'tune_frequency': 10000}

    #     # basic mcmc run with normal normal
    #     mcmc_run = MCMC(MH(model, NormalProposal(**mh_params)), num_samples=2000, warmup_steps=350)  # , lag_steps=5)

    #     t_values = []
    #     for trace, _ in mcmc_run._traces():
    #         t_values += [trace.node['mu_latent']['value']]

    #     # marginal = Marginal(mcmc_run)
    #     # samples = []
    #     # for _ in range(400):
    #     #     samples.append(marginal.sample())

    #     sample_mean = torch.mean(torch.stack(t_values), 0)
    #     # sample_std = torch.std(torch.stack(samples), 0)
    #     print("Mean : {}, analytic: {}".format(sample_mean, self.analytic_mu_n))
    #     bb()
    #     self.assertEqual(0.0, param_mse(sample_mean, self.analytic_mu_n), prec=0.05)

        # adam = optim.Adam({"lr": .001})
        # svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)
        # for k in range(n_steps):
        #     svi.step()

        #     mu_error = param_mse("mu_q", self.analytic_mu_n)
        #     log_sig_error = param_mse("log_sig_q", self.analytic_log_sig_n)

        # bb()
        # self.assertEqual(0.0, log_sig_error, prec=0.05)

# class GaussianChain(object):

#     def __init__(self, dim, chain_len, num_obs):
#         self.dim = dim
#         self.chain_len = chain_len
#         self.num_obs = num_obs
#         self.mu_0 = Variable(torch.Tensor(torch.zeros(self.dim)), requires_grad=True)
#         self.lambda_prec = Variable(torch.Tensor(torch.ones(self.dim)))

#     def model(self, data):
#         mu = pyro.param('mu_0', self.mu_0)
#         lambda_prec = self.lambda_prec
#         for i in range(1, self.chain_len + 1):
#             mu = pyro.sample('mu_{}'.format(i), dist.normal, mu=mu, sigma=Variable(lambda_prec.data))
#         pyro.sample('obs', dist.normal, mu=mu, sigma=Variable(lambda_prec.data), obs=data)

#     @property
#     def data(self):
#         return Variable(torch.ones(self.num_obs, self.dim))

#     def id_fn(self):
#         return 'dim={}_chain-len={}_num_obs={}'.format(self.dim, self.chain_len, self.num_obs)


# def normal_normal_model(data):
#     x = pyro.param('mu', Variable(torch.Tensor([0.0])))
#     y = pyro.sample('x', dist.normal, mu=x, sigma=Variable(torch.Tensor([1])))
#     pyro.sample('obs', dist.normal, mu=y, sigma=Variable(torch.Tensor([1])), obs=data)
#     return y


# def rmse(t1, t2):
#     return (t1 - t2).pow(2).mean().sqrt()


# T = namedtuple('TestCase', ['fixture', 'mh_params', 'mcmc_params', 'expected_means', 'expected_precs', 'mean_tol', 'std_tol'])

# # TODO: delete slow test cases once code is in master
# TEST_CASES = [
#     T(GaussianChain(dim=10, chain_len=3, num_obs=1),
#       mh_params={'mu': ng_zeros(1), 'sigma': ng_ones(1), 'tune_frequency': 100},
#       mcmc_params={'num_samples': 800,
#                    'warmup_steps': 50},
#       expected_means=[0.25, 0.50, 0.75],
#       expected_precs=[1.33, 1, 1.33],
#       mean_tol=0.03,
#       std_tol=0.06),
#     # XXX: Very sensitive to HMC parameters. Biased estimate is obtained
#     # without enough samples and/or larger step size.
#     T(GaussianChain(dim=5, chain_len=2, num_obs=10000),
#       mh_params={'mu': ng_zeros(1), 'sigma': ng_ones(1), 'tune_frequency': 100},
#       mcmc_params={'num_samples': 2000,
#                    'warmup_steps': 500},
#       expected_means=[0.5, 1.0],
#       expected_precs=[2.0, 10000],
#       mean_tol=0.05,
#       std_tol=0.05),
#     T(GaussianChain(dim=10, chain_len=4, num_obs=1),
#       mh_params={'mu': ng_zeros(1), 'sigma': ng_ones(1), 'tune_frequency': 100},
#       mcmc_params={'num_samples': 1200,
#                    'warmup_steps': 300},
#       expected_means=[0.20, 0.40, 0.60, 0.80],
#       expected_precs=[1.25, 0.83, 0.83, 1.25],
#       mean_tol=0.06,
#       std_tol=0.06),
#     T(GaussianChain(dim=5, chain_len=9, num_obs=1),
#       mh_params={'mu': ng_zeros(1), 'sigma': ng_ones(1), 'tune_frequency': 100},
#       mcmc_params={'num_samples': 3000,
#                    'warmup_steps': 500},
#       expected_means=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
#       expected_precs=[1.11, 0.63, 0.48, 0.42, 0.4, 0.42, 0.48, 0.63, 1.11],
#       mean_tol=0.08,
#       std_tol=0.08),
# ]

# TEST_IDS = [t[0].id_fn() for t in TEST_CASES]


# @pytest.mark.parametrize(
#     'fixture, mh_params, mcmc_params, expected_means, expected_precs, mean_tol, std_tol', TEST_CASES, ids=TEST_IDS)
# def test_mh_conjugate_gaussian(fixture, mh_params, mcmc_params, expected_means, expected_precs, mean_tol, std_tol):

    # mcmc_run = MCMC(MH(fixture.model, NormalProposal(**mh_params)), **mcmc_params)
#     pyro.get_param_store().clear()
#     post_trace = defaultdict(list)
#     for t, _ in mcmc_run._traces(fixture.data):
#         for i in range(1, fixture.chain_len + 1):
#             param_name = 'mu_' + str(i)
#             post_trace[param_name].append(t.nodes[param_name]['value'])
#     logger.info('Acceptance ratio: {}'.format(mcmc_run.kernel.acceptance_ratio))
#     for i in range(1, fixture.chain_len + 1):
#         param_name = 'mu_' + str(i)
#         latent_mu = torch.mean(torch.stack(post_trace[param_name]), 0)
#         latent_std = torch.std(torch.stack(post_trace[param_name]), 0)
#         expected_mean = Variable(torch.ones(fixture.dim) * expected_means[i - 1])
#         expected_std = 1 / torch.sqrt(Variable(torch.ones(fixture.dim) * expected_precs[i - 1]))

#         # Actual vs expected posterior means for the latents
#         logger.info('Posterior mean (actual) - {}'.format(param_name))
#         logger.info(latent_mu)
#         logger.info('Posterior mean (expected) - {}'.format(param_name))
#         logger.info(expected_mean)
#         assert_equal(rmse(latent_mu, expected_mean).data[0], 0, prec=mean_tol)

#         # Actual vs expected posterior precisions for the latents
#         logger.info('Posterior std (actual) - {}'.format(param_name))
#         logger.info(latent_std)
#         logger.info('Posterior std (expected) - {}'.format(param_name))
#         logger.info(expected_std)
#         assert_equal(rmse(latent_std, expected_std).data[0], 0, prec=std_tol)


# def test_mh_interface():

#     mh_params = {'mu': ng_zeros(1), 'sigma': ng_ones(1), 'tune_frequency': 50}

#     # basic mcmc run with normal normal
#     mcmc_run = MCMC(MH(normal_normal_model, NormalProposal(**mh_params)), num_samples=800, warmup_steps=75)

#     data = Variable(torch.Tensor([1.0]))
#     marginal = Marginal(mcmc_run)
#     samples = []
#     for _ in range(400):
#         samples.append(marginal.sample(data))
#     sample_mean = torch.mean(torch.stack(samples), 0)
#     sample_std = torch.std(torch.stack(samples), 0)
#     # bb()
#     assert_equal(sample_mean.data, torch.Tensor([0.0]), prec=5e-2)
#     assert_equal(sample_std.data, torch.Tensor([1.0]), prec=5e-2)


if __name__ == "__main__":
    from pdb import set_trace as bb

    # generic test
    nnt = MHTest()
    nnt.setUp()
    nnt.test_single_site_mh()

    # # test_mh_interface()
    # raise NotImplementedError()

    # # more complicated test
    # tf = T(GaussianChain(dim=10, chain_len=3, num_obs=1),
    #        mh_params={'mu': ng_zeros(1), 'sigma': ng_ones(1), 'tune_frequency': 100},
    #        mcmc_params={'num_samples': 1200,
    #                     'warmup_steps': 300},
    #        expected_means=[0.25, 0.50, 0.75],
    #        expected_precs=[1.33, 1, 1.33],
    #        mean_tol=0.03,
    #        std_tol=0.06)

    # test_mh_conjugate_gaussian(**{field: getattr(tf, field) for field in tf._fields})
