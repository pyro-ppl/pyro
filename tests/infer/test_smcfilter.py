import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SMCFilter


class SmokeModel:

    def __init__(self, state_size, plate_size):
        self.state_size = state_size
        self.plate_size = plate_size

    def init(self):
        self.t = 0
        self.x_mean = pyro.sample("x_mean", dist.Normal(0., 1.))
        self.y_mean = pyro.sample("y_mean",
                                  dist.MultivariateNormal(torch.zeros(self.state_size), torch.eye(self.state_size)))

    def step(self, x=None, y=None):
        v = pyro.sample("v_{}".format(self.t), dist.Normal(0., 1.))
        with pyro.plate("plate", self.plate_size):
            w = pyro.sample("w_{}".format(self.t), dist.Normal(v, 1.))
            x = pyro.sample("x_{}".format(self.t),
                            dist.Normal(self.x_mean + w, 1), obs=x)
            y = pyro.sample("y_{}".format(self.t),
                            dist.MultivariateNormal(self.y_mean + w.unsqueeze(-1), torch.eye(self.state_size)),
                            obs=y)
        self.t += 1

        return x, y


class SmokeGuide:

    def __init__(self, state_size, plate_size):
        self.state_size = state_size
        self.plate_size = plate_size

    def init(self):
        self.t = 0
        self.x_mean = pyro.sample("x_mean",
                                  dist.Normal(0., 2.))
        self.y_mean = pyro.sample("y_mean",
                                  dist.MultivariateNormal(torch.zeros(self.state_size), 2.*torch.eye(self.state_size)))

    def step(self, x=None, y=None):
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

    true_model.init()
    truth = [true_model.step() for t in range(num_steps)]

    smc.init()
    for xy in truth:
        smc.step(*xy)
    smc.get_values_and_log_weights()
    smc.get_empirical()


class HarmonicModel:

    def __init__(self):
        self.A = torch.tensor([[0., 1.],
                               [-1., 0.]])
        self.B = torch.tensor([3., 3.])
        self.sigma_z = torch.tensor(1.)
        self.sigma_y = torch.tensor(1.)

    def init(self):
        self.t = 0
        self.z = torch.tensor([1., 0.])
        self.y = None

    def step(self, y=None):
        self.t += 1
        self.z = pyro.sample("z_{}".format(self.t),
                             dist.Normal(self.z.matmul(self.A), self.B*self.sigma_z).to_event(1))
        self.y = pyro.sample("y_{}".format(self.t),
                             dist.Normal(self.z[..., 0], self.sigma_y),
                             obs=y)

        return self.z, self.y


class HarmonicGuide:

    def __init__(self):
        self.model = HarmonicModel()

    def init(self):
        self.t = 0
        self.z = torch.tensor([1., 0.])

    def step(self, y=None):
        self.t += 1

        # Proposal distribution
        self.z = pyro.sample("z_{}".format(self.t),
                             dist.Normal(self.z.matmul(self.model.A), torch.tensor([2., 2.])).to_event(1))


def generate_data():
    model = HarmonicModel()

    model.init()
    zs = [torch.tensor([1., 0.])]
    ys = [None]
    for t in range(50):
        z, y = model.step()
        zs.append(z)
        ys.append(y)

    return zs, ys


def score_latent(zs, ys):
    model = HarmonicModel()
    with poutine.trace() as trace:
        with poutine.condition(data={"z_{}".format(t): z for t, z in enumerate(zs)}):
            model.init()
            for y in ys[1:]:
                model.step(y)

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
    values, logweights = smc.get_values_and_log_weights()
    i = logweights.max(0)[1]
    values = {k: v[i] for k, v in values.items()}

    zs_pred = [torch.tensor([1., 0.])]
    zs_pred += [values.get("z_{}".format(t)) for t in range(1, 51)]

    assert(score_latent(zs_true, ys_true) > score_latent(zs, ys_true))
    assert(score_latent(zs_pred, ys_true) > score_latent(zs_pred, ys))
    assert(score_latent(zs_pred, ys_true) > score_latent(zs, ys_true))
