import torch
from torch import nn
from torch.distributions import constraints

import pyro.distributions.modules as mist
from pyro.nn.module import PyroModule, PyroParam


def test_smoke():
    module = PyroModule()
    module.a = mist.Normal(nn.Parameter(torch.tensor(0.)), 1)
    module.b = mist.MultivariateNormal(
        torch.zeros(3),
        scale_tril=PyroParam(torch.eye(3), constraint=constraints.lower_cholesky))
