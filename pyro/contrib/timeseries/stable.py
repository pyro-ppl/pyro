# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import torch
from torch import nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, init_to_feasible
from pyro.infer.reparam import ConjugateReparam, LinearHMMReparam, StableReparam
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan

logger = logging.getLogger(__name__)


def bounded_exp(x, bound):
    return (x - math.log(bound)).sigmoid() * bound


class StableModel(PyroModule):
    def __init__(self, name, hidden_dim, obs_dim, max_rate):
        assert isinstance(name, str)
        assert isinstance(hidden_dim, int) and hidden_dim >= 1
        assert isinstance(obs_dim, int) and obs_dim >= 1
        assert isinstance(max_rate, (int, float)) and max_rate >= 1
        self.name = name
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.max_rate = max_rate
        super().__init__()

        # All of the following can be overridden in instances or subclasses.
        self.stability = PyroSample(dist.Uniform(1, 2))
        self.init_scale = torch.full([hidden_dim], max_rate)
        self.init_skew = 0.
        self.init_loc = 0.
        self.trans_matrix = PyroParam(lambda: torch.eye(hidden_dim))
        self.trans_skew = PyroSample(dist.Uniform(-1, 1).expand([hidden_dim]).to_event(1))
        self.trans_scale = PyroSample(dist.LogNormal(0, 10).expand([hidden_dim]).to_event(1))
        self.trans_loc = PyroSample(dist.Normal(0, 10).expand([hidden_dim]).to_event(1))
        self.obs_matrix = PyroParam(lambda: torch.eye(hidden_dim, obs_dim))
        self.obs_skew = PyroSample(dist.Uniform(-1, 1).expand([obs_dim]).to_event(1))
        self.obs_scale = PyroSample(dist.LogNormal(0, 10).expand([obs_dim]).to_event(1))
        self.obs_loc = PyroSample(dist.Normal(0, 10).expand([obs_dim]).to_event(1))

    def forward(self, data=None):
        init_dist = dist.Stable(self.stability, self.init_skew, self.init_scale, self.init_loc)
        trans_dist = dist.Stable(self.stability, self.trans_skew, self.trans_scale, self.trans_loc)
        obs_dist = dist.Stable(self.stability, self.obs_skew, self.obs_scale, self.obs_loc)
        hmm = dist.LinearHMM(init_dist.to_event(1),
                             self.trans_matrix, trans_dist.to_event(1),
                             self.obs_matrix, obs_dist.to_event(1))
        self.hmm = hmm
        log_rate = pyro.sample(self.name, hmm)
        rate = bounded_exp(log_rate, self.max_rate)
        return pyro.sample("{}_counts".format(self.name), dist.Poisson(rate).to_event(2), obs=data)


class LogNormalCoxGuide:
    """
    Expectation propagation approximate update for a Poisson Likelihood with
    log-normal prior.
    """
    def __init__(self, event_dim):
        self.event_dim = event_dim

    def __call__(self, data):
        # See https://en.wikipedia.org/wiki/Gamma_distribution#Logarithmic_expectation_and_variance
        data = data.clamp(min=0.5)  # Work around asymptote at data==0.
        loc = data.digamma()
        scale = 1.2825498301618641  # = sqrt(trigamma(1))
        return dist.Normal(loc, scale).to_event(self.event_dim)


class LogStableCoxProcess(nn.Module):
    def __init__(self, name, hidden_dim, obs_dim, max_rate):
        super().__init__()
        model = StableModel(name, hidden_dim, obs_dim, max_rate)
        rep = StableReparam()
        # FIXME poutine-wrapped PyroModules are not nn.Modules.
        model = poutine.reparam(model, {name: LinearHMMReparam(rep, rep, rep)})
        model = poutine.reparam(model, {name: ConjugateReparam(LogNormalCoxGuide(event_dim=2))})
        self.model = model
        self.guide = AutoDiagonalNormal(model, init_loc_fn=init_to_feasible)

    def fit(self, data,
            num_steps=100,
            learning_rate=1e-2):
        optim = ClippedAdam({"lr": learning_rate, "betas": (0.8, 0.95)})
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, elbo)
        losses = []
        for step in range(num_steps):
            loss = svi.step(data)
            logger.info("step {: >4d} loss = {:0.4g}".format(step, loss))
            assert not torch_isnan(loss)
            losses.append(loss)
        return losses

    @torch.no_grad()
    def detect(self, data):
        """
        Extract normalized noise vectors.
        These vectors can be sorted by magnitude to extract detections.
        """
        guide_trace = poutine.trace(self.guide).get_trace(data)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(data)

        hmm = self.model.hmm
        x = model_trace.nodes[self.name]["value"]
        z = x._pyro_latent

        z_pred = z @ hmm.transition_matrix - hmm.transition_dist.mean
        trans = (z[..., 1:, :] - z_pred[..., :-1, :]) / hmm.transition_dist.scale

        x_pred = z @ hmm.observation_matrix - hmm.observation_dist.mean
        obs = (x - x_pred) / hmm.observation_dist.scale

        return {"trans": trans, "obs": obs}
