import logging
import math
from unittest import TestCase

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.distributions.util import logsumexp
from pyro.infer import TraceEnum_ELBO, SVI, config_enumerate, Trace_ELBO, TracePredictive, TracePosterior
import pyro.optim as optim
import pyro.poutine as poutine
from tests.common import assert_equal


logging.basicConfig(format='%(message)s', level=logging.INFO)


class DiscreteDag(TestCase):
    """
    . represents direction of the dependency.

    a --. b
    |  /  |
    ..    .
    c -- .d --. e
          |  /
          . .
          f
    """

    def setUp(self):
        self.params = {
            "cpd_a": torch.tensor([0.15, 0.85]),
            "cpd_b": torch.tensor([[0.3, 0.7], [0.8, 0.2]]),
            "cpd_c": torch.tensor([[[0.2, 0.8], [0.4, 0.6]], [[0.33, 0.67], [0.22, 0.78]]]),
            "cpd_d": torch.tensor([[[0.2, 0.8], [0.1, 0.9]], [[0.67, 0.33], [0.25, 0.75]]]),
            "cpd_e": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            "cpd_f": torch.tensor([[[0.25, 0.75], [0.8, 0.2]], [[0.3, 0.7], [0.9, 0.1]]])
        }

    @staticmethod
    @poutine.broadcast
    @config_enumerate(default="parallel", expand=False)
    def model(N, data=None):
        cpd_a = pyro.param("cpd_a", torch.rand(2), constraint=constraints.simplex)
        cpd_b = pyro.param("cpd_b", torch.rand(2, 2), constraint=constraints.simplex)
        cpd_c = pyro.param("cpd_c", torch.rand(2, 2, 2), constraint=constraints.simplex)
        cpd_d = pyro.param("cpd_d", torch.rand(2, 2, 2), constraint=constraints.simplex)
        cpd_e = pyro.param("cpd_e", torch.rand(2, 2), constraint=constraints.simplex)
        cpd_f = pyro.param("cpd_f", torch.rand(2, 2, 2), constraint=constraints.simplex)
        with pyro.iarange("data", N):
            a = pyro.sample("a", dist.Categorical(cpd_a))
            b = pyro.sample("b", dist.Categorical(cpd_b[a]))
            c = pyro.sample("c", dist.Categorical(cpd_c[a, b]))
            d = pyro.sample("d", dist.Categorical(cpd_d[b, c]))
            e = pyro.sample("e", dist.Categorical(cpd_e[d]))
            pyro.sample("f", dist.Categorical(cpd_f[d, e]))

    @staticmethod
    @poutine.broadcast
    @config_enumerate(default="parallel", expand=False)
    def null_guide(N, data):
        pass

    @staticmethod
    @poutine.broadcast
    @config_enumerate(default="parallel", expand=False)
    def guide_partially_observed(N, data):
        cpd_e = pyro.param("cpd_e", torch.ones(2, 2) / 2., constraint=constraints.simplex)
        with pyro.iarange("data", N, dim=-1):
            d = data[:, 3].long()
            pyro.sample("e", dist.Categorical(cpd_e[d]))

    def model_fully_observed(self, N, data):
        data = {"a": data[:, 0], "b": data[:, 1], "c": data[:, 2],
                "d": data[:, 3], "e": data[:, 4], "f": data[:, 5]}
        return poutine.condition(self.model, data=data)(N, data)

    def model_partially_observed(self, N, data):
        data = {"a": data[:, 0], "b": data[:, 1], "c": data[:, 2],
                "d": data[:, 3], "f": data[:, 5]}
        return poutine.condition(self.model, data=data)(N, data)

    def generate_data(self, n=1000):
        pyro.get_param_store().set_state({"params": self.params,
                                          "constraints": {k: constraints.real for k in self.params.keys()}})
        trace = poutine.trace(self.model).get_trace(n)
        data = torch.stack([trace.nodes["a"]["value"],
                            trace.nodes["b"]["value"],
                            trace.nodes["c"]["value"],
                            trace.nodes["d"]["value"],
                            trace.nodes["e"]["value"],
                            trace.nodes["f"]["value"]
                            ], dim=-1)
        return data

    @staticmethod
    def log_predictive_density(model, model_trace_posterior, data):
        test_eval = TracePredictive(model,
                                    model_trace_posterior,
                                    num_samples=1000)
        test_eval.run(len(data), data)
        trace_log_pdf = []
        for tr in test_eval.exec_traces:
            trace_log_pdf.append(tr.log_prob_sum())
        return logsumexp(torch.stack(trace_log_pdf), dim=-1) - math.log(len(trace_log_pdf))

    @staticmethod
    def run_svi(model, guide, loss, data, batch_size=128):
        pyro.clear_param_store()
        adam = optim.Adam({"lr": 0.001 * batch_size / 128, "betas": (0.95, 0.999)})
        svi = SVI(model, guide, adam, loss=loss, num_samples=1000)
        num_epochs = 50 + 2 * batch_size // 128
        for n in range(num_epochs):
            for i in range(len(data) // batch_size):
                loss = svi.step(batch_size, data[i: i + batch_size])
            logging.info("Epoch[{}] Elbo loss: {}".format(n, loss))
        return svi

    def test_fully_observed(self):
        pyro.clear_param_store()
        data = self.generate_data(n=10000)
        self.run_svi(self.model_fully_observed, self.null_guide,
                     Trace_ELBO(max_iarange_nesting=1), data, batch_size=len(data))
        for k, v in self.params.items():
            assert_equal(pyro.param(k), v, prec=0.05)

    def test_partially_observed(self):
        pyro.clear_param_store()
        data = self.generate_data(n=10000)
        svi_partially_obs = self.run_svi(self.model_partially_observed, self.guide_partially_observed,
                                         TraceEnum_ELBO(max_iarange_nesting=1), data)

        partially_obs_posterior = svi_partially_obs.run(len(data), data)
        partially_obs_log_pred = self.log_predictive_density(self.model_partially_observed,
                                                             partially_obs_posterior,
                                                             data)
        for p in self.params:
            logging.info("{}: {}".format(p, pyro.param(p)))
        svi_fully_obs = self.run_svi(self.model_fully_observed, self.null_guide,
                                     Trace_ELBO(max_iarange_nesting=1), data, batch_size=len(data))
        fully_obs_posterior = svi_fully_obs.run(len(data), data)
        fully_obs_log_pred = self.log_predictive_density(self.model_fully_observed,
                                                         fully_obs_posterior,
                                                         data)
        # Check log likelihood rather than exact parameter equality
        assert abs(fully_obs_log_pred - partially_obs_log_pred) / fully_obs_log_pred < 0.05
