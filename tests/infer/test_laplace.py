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
    # with normal prior, check mu_hess = 1 + len(data)
    assert_equal(mu_hess[0, 0], 4.0, prec=0.1)


def test_laplace_param():

    def model(data):
        mu = pyro.param("mu", Variable(torch.zeros(1), requires_grad=True))
        for i, x in enumerate(data):
            pyro.observe("x_{}".format(i), dist.normal, x, mu, ng_ones(1))

    data = Variable(torch.Tensor([0, 1, 3, 4]))
    optim = Adam({"lr": 1e-1})
    infer = Laplace(model, optim)
    for _ in range(100):
        infer.step(data)

    mu_hess = infer.get_hessians(["mu"], data)["mu"]
    # with uniform prior, check mu_hess = len(data)
    assert_equal(mu_hess[0, 0], 4.0, prec=0.1)


def test_laplace_chain_rule():

    def model():
        x = pyro.sample("x", dist.normal, ng_zeros(1), ng_ones(1))
        y = pyro.sample("y", dist.normal, x, ng_ones(1))
        pyro.observe("z", dist.normal, y, ng_ones(1) * 3.0, ng_ones(1))

    optim = Adam({"lr": 1e-1})
    infer = Laplace(model, optim)
    for _ in range(100):
        infer.step()

    hess = infer.get_hessians(["x", "y"])
    assert_equal(hess["x"][0, 0], 2.0, prec=0.1)
    assert_equal(hess["y"][0, 0], 2.0, prec=0.1)
