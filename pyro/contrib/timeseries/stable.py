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
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan

logger = logging.getLogger(__name__)


def bounded_exp(x, bound):
    return (x - math.log(bound)).sigmoid() * bound


class StableModel(PyroModule):
    def __init__(self, name, hidden_dim, obs_dim):
        assert isinstance(name, str)
        self.name = name
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        super().__init__()

    def forward(self, data=None):
        hidden_dim = self.hidden_dim
        obs_dim = self.obs_dim
        stability = pyro.sample("stability", dist.Uniform(0, 2))
        init_scale = 1e6
        trans_matrix = "TODO"
        trans_skew = pyro.sample("trans_skew", dist.Uniform(-1, 1).expand([hidden_dim]).to_event(1))
        trans_scale = pyro.sample("trans_scale", dist.LogNormal(0, 10).expand([hidden_dim]).to_event(1))
        trans_loc = pyro.sample("trans_loc", dist.Normal(0, 10).expand([hidden_dim]).to_event(1))
        obs_matrix = "TODO"
        obs_skew = pyro.sample("obs_skew", dist.Uniform(-1, 1).expand([obs_dim]).to_event(1))
        obs_scale = pyro.sample("obs_scale", dist.LogNormal(0, 10).expand([obs_dim]).to_event(1))
        obs_loc = pyro.sample("obs_loc", dist.Normal(0, 10).expand([obs_dim]).to_event(1))
        hmm = dist.LinearHMM(dist.Stable(stability, 0, init_scale).to_event(1),
                             trans_matrix,
                             dist.Stable(stability, trans_skew, trans_scale, trans_loc).to_event(1),
                             obs_matrix,
                             dist.Stable(stability, obs_skew, obs_scale, obs_loc).to_event(1))
        self.hmm = hmm
        return pyro.sample(self.name, hmm, obs=data)


class LogNormalCoxGuide:
    """
    Expectation propagation approximate update for a Poisson Likelihood with
    log-normal prior.
    """
    def __init__(self, event_dim):
        one = torch.tensor(1., dtype=torch.double, device="cpu")
        self.digamma_one = one.digamma().item()
        self.scale = one.polygamma(1).sqrt().item()
        self.event_dim = event_dim

    def __call__(self, data):
        # See https://en.wikipedia.org/wiki/Gamma_distribution#Logarithmic_expectation_and_variance
        loc = data.digamma() - self.digamma_one
        return dist.Normal(loc, self.scale).to_event(self.event_dim)


class LogStableCoxProcess(nn.Module):
    """
    """
    def __init__(self, name):
        model = StableModel(name)
        rep = StableReparam()
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
