import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer.laplace import Laplace
from pyro.optim import Adam
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal


def test_laplace_sample():

    def model(data):
        mu = pyro.sample("mu", dist.normal, ng_zeros(1), ng_ones(1))
        for i, x in enumerate(data):
            pyro.observe("x_{}".format(i), dist.normal, x, mu, ng_ones(1))

    data = Variable(torch.Tensor([1, 3, 4]))
    optim = Adam({"lr": 1e-1})
    infer = Laplace(model, optim)
    for _ in range(100):
        infer.step(data)

    mu_hess = infer.get_hessians(["mu"], data)["mu"]
    assert_equal(mu_hess[0, 0], 4, prec=0.1)
