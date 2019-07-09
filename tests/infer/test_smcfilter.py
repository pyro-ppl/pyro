from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SMCFilter


class Model:

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


class Guide:

    def __init__(self):
        self.model = Model()

    def init(self):
        self.t = 0
        self.z = torch.tensor([1., 0.])

    def step(self, y=None):
        self.t += 1

        # Proposal distribution
        self.z = pyro.sample("z_{}".format(self.t),
                             dist.Normal(self.z.matmul(self.model.A), torch.tensor([2., 2.])).to_event(1))


def generate_data():
    model = Model()

    model.init()
    zs = [torch.tensor([1., 0.])]
    ys = [None]
    for t in range(50):
        z, y = model.step()
        zs.append(z)
        ys.append(y)

    return zs, ys


def score_latent(zs, ys):
    model = Model()
    with poutine.trace() as trace:
        with poutine.condition(data={"z_{}".format(t): z for t, z in enumerate(zs)}):
            model.init()
            for y in ys[1:]:
                model.step(y)

    return trace.trace.log_prob_sum()


def test_likelihood_ratio():

    model = Model()
    guide = Guide()

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

    for t in range(51):
        print("{}\t{}\t{}\t{}".format(t, zs[t], zs_true[t], zs_pred[t]))
    assert(score_latent(zs_true, ys_true) > score_latent(zs, ys_true))
    assert(score_latent(zs_pred, ys_true) > score_latent(zs_pred, ys))
    assert(score_latent(zs_pred, ys_true) > score_latent(zs, ys_true))
