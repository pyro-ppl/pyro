import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.nn.module import PyroModule, pyro_param, pyro_sample
from pyro.optim import Adam


class Model(PyroModule):
    def __init__(self):
        super().__init__()
        self.loc = pyro_param(torch.zeros(2))
        self.scale = pyro_param(torch.ones(2), constraint=constraints.positive)
        self.z = pyro_sample(lambda self: dist.Normal(self.loc, self.scale).to_event(1))

    def forward(self, data):
        loc, log_scale = self.z.unbind(-1)
        with pyro.plate("data"):
            pyro.sample("obs", dist.Cauchy(loc, log_scale.exp()),
                        obs=data)


class Guide(PyroModule):
    def __init__(self):
        super().__init__()
        self.loc = pyro_param(torch.zeros(2))
        self.scale = pyro_param(torch.ones(2), constraint=constraints.positive)
        self.z = pyro_sample(lambda self: dist.Normal(self.loc, self.scale).to_event(1))

    def forward(self, *args, **kwargs):
        return self.z


def test_svi_smoke():
    data = torch.randn(5)
    model = Model()
    guide = Guide()

    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, Trace_ELBO())
    for step in range(3):
        svi.step(data)
