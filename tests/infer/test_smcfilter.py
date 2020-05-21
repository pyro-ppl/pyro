# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SMCFilter
from pyro.infer.smcfilter import _systematic_sample
from tests.common import assert_close


@pytest.mark.parametrize("size", range(1, 32))
def test_systematic_sample(size):
    pyro.set_rng_seed(size)
    probs = torch.randn(size).exp()
    probs /= probs.sum()

    num_samples = 20000
    index = _systematic_sample(probs.expand(num_samples, size))
    histogram = torch.zeros_like(probs)
    histogram.scatter_add_(-1, index.reshape(-1),
                           probs.new_ones(1).expand(num_samples * size))

    expected = probs * size
    actual = histogram / num_samples
    assert_close(actual, expected, atol=0.01)


class SmokeModel:

    def __init__(self, state_size, plate_size):
        self.state_size = state_size
        self.plate_size = plate_size

    def init(self, state):
        self.t = 0
        state["x_mean"] = pyro.sample("x_mean", dist.Normal(0., 1.))
        state["y_mean"] = pyro.sample("y_mean",
                                      dist.MultivariateNormal(torch.zeros(self.state_size),
                                                              torch.eye(self.state_size)))

    def step(self, state, x=None, y=None):
        v = pyro.sample("v_{}".format(self.t), dist.Normal(0., 1.))
        with pyro.plate("plate", self.plate_size):
            w = pyro.sample("w_{}".format(self.t), dist.Normal(v, 1.))
            x = pyro.sample("x_{}".format(self.t),
                            dist.Normal(state["x_mean"] + w, 1), obs=x)
            y = pyro.sample("y_{}".format(self.t),
                            dist.MultivariateNormal(state["y_mean"] + w.unsqueeze(-1), torch.eye(self.state_size)),
                            obs=y)
        self.t += 1
        return x, y


class SmokeGuide:

    def __init__(self, state_size, plate_size):
        self.state_size = state_size
        self.plate_size = plate_size

    def init(self, state):
        self.t = 0
        pyro.sample("x_mean", dist.Normal(0., 2.))
        pyro.sample("y_mean",
                    dist.MultivariateNormal(torch.zeros(self.state_size),
                                            2.*torch.eye(self.state_size)))

    def step(self, state, x=None, y=None):
        v = pyro.sample("v_{}".format(self.t), dist.Normal(0., 2.))
        with pyro.plate("plate", self.plate_size):
            pyro.sample("w_{}".format(self.t), dist.Normal(v, 2.))
        self.t += 1


@pytest.mark.parametrize("max_plate_nesting", [1, 2])
@pytest.mark.parametrize("state_size", [2, 5, 1])
@pytest.mark.parametrize("plate_size", [3, 7, 1])
@pytest.mark.parametrize("num_steps", [1, 2, 10])
def test_smoke(max_plate_nesting, state_size, plate_size, num_steps):
    model = SmokeModel(state_size, plate_size)
    guide = SmokeGuide(state_size, plate_size)

    smc = SMCFilter(model, guide, num_particles=100, max_plate_nesting=max_plate_nesting)

    true_model = SmokeModel(state_size, plate_size)

    state = {}
    true_model.init(state)
    truth = [true_model.step(state) for t in range(num_steps)]

    smc.init()
    assert set(smc.state) == {"x_mean", "y_mean"}
    for x, y in truth:
        smc.step(x, y)
    assert set(smc.state) == {"x_mean", "y_mean"}
    smc.get_empirical()


class HarmonicModel:

    def __init__(self):
        self.A = torch.tensor([[0., 1.],
                               [-1., 0.]])
        self.B = torch.tensor([3., 3.])
        self.sigma_z = torch.tensor(1.)
        self.sigma_y = torch.tensor(1.)

    def init(self, state):
        self.t = 0
        state["z"] = pyro.sample("z_init",
                                 dist.Delta(torch.tensor([1., 0.]), event_dim=1))

    def step(self, state, y=None):
        self.t += 1
        state["z"] = pyro.sample("z_{}".format(self.t),
                                 dist.Normal(state["z"].matmul(self.A),
                                             self.B*self.sigma_z).to_event(1))
        y = pyro.sample("y_{}".format(self.t),
                        dist.Normal(state["z"][..., 0], self.sigma_y),
                        obs=y)

        state["z_{}".format(self.t)] = state["z"]  # saved for testing

        return state["z"], y


class HarmonicGuide:

    def __init__(self):
        self.model = HarmonicModel()

    def init(self, state):
        self.t = 0
        pyro.sample("z_init", dist.Delta(torch.tensor([1., 0.]), event_dim=1))

    def step(self, state, y=None):
        self.t += 1

        # Proposal distribution
        pyro.sample("z_{}".format(self.t),
                    dist.Normal(state["z"].matmul(self.model.A),
                                torch.tensor([2., 2.])).to_event(1))


def generate_data():
    model = HarmonicModel()

    state = {}
    model.init(state)
    zs = [torch.tensor([1., 0.])]
    ys = [None]
    for t in range(50):
        z, y = model.step(state)
        zs.append(z)
        ys.append(y)

    return zs, ys


def score_latent(zs, ys):
    model = HarmonicModel()
    with poutine.trace() as trace:
        with poutine.condition(data={"z_{}".format(t): z for t, z in enumerate(zs)}):
            state = {}
            model.init(state)
            for y in ys[1:]:
                model.step(state, y)

    return trace.trace.log_prob_sum()


def test_likelihood_ratio():

    model = HarmonicModel()
    guide = HarmonicGuide()

    smc = SMCFilter(model, guide, num_particles=100, max_plate_nesting=0)

    zs, ys = generate_data()
    zs_true, ys_true = generate_data()
    smc.init()
    for y in ys_true[1:]:
        smc.step(y)
    i = smc.state._log_weights.max(0)[1]
    values = {k: v[i] for k, v in smc.state.items()}

    zs_pred = [torch.tensor([1., 0.])]
    zs_pred += [values["z_{}".format(t)] for t in range(1, 51)]

    assert(score_latent(zs_true, ys_true) > score_latent(zs, ys_true))
    assert(score_latent(zs_pred, ys_true) > score_latent(zs_pred, ys))
    assert(score_latent(zs_pred, ys_true) > score_latent(zs, ys_true))


def test_gaussian_filter():
    dim = 4
    init_dist = dist.MultivariateNormal(torch.zeros(dim), scale_tril=torch.eye(dim) * 10)
    trans_mat = torch.eye(dim)
    trans_dist = dist.MultivariateNormal(torch.zeros(dim), scale_tril=torch.eye(dim))
    obs_mat = torch.eye(dim)
    obs_dist = dist.MultivariateNormal(torch.zeros(dim), scale_tril=torch.eye(dim) * 2)
    hmm = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)

    class Model:
        def init(self, state):
            state["z"] = pyro.sample("z_init", init_dist)
            self.t = 0

        def step(self, state, datum=None):
            state["z"] = pyro.sample("z_{}".format(self.t),
                                     dist.MultivariateNormal(state["z"], scale_tril=trans_dist.scale_tril))
            datum = pyro.sample("obs_{}".format(self.t),
                                dist.MultivariateNormal(state["z"], scale_tril=obs_dist.scale_tril),
                                obs=datum)
            self.t += 1
            return datum

    class Guide:
        def init(self, state):
            pyro.sample("z_init", init_dist)
            self.t = 0

        def step(self, state, datum):
            pyro.sample("z_{}".format(self.t),
                        dist.MultivariateNormal(state["z"], scale_tril=trans_dist.scale_tril * 2))
            self.t += 1

    # Generate data.
    num_steps = 20
    model = Model()
    state = {}
    model.init(state)
    data = torch.stack([model.step(state) for _ in range(num_steps)])

    # Perform inference.
    model = Model()
    guide = Guide()
    smc = SMCFilter(model, guide, num_particles=1000, max_plate_nesting=0)
    smc.init()
    for t, datum in enumerate(data):
        smc.step(datum)
        expected = hmm.filter(data[:1+t])
        actual = smc.get_empirical()["z"]
        assert_close(actual.variance ** 0.5, expected.variance ** 0.5, atol=0.1, rtol=0.5)
        sigma = actual.variance.max().item() ** 0.5
        assert_close(actual.mean, expected.mean, atol=3 * sigma)
