# This file tests a variety of model,guide pairs with valid and invalid syntax.

import pytest

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI


def test_nested_irange_ok():

    def model():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange-1", 100, 10):
            for j in pyro.irange("irange-2", 100, 10):
                pyro.sample("x_{}_{}".format(i, j), dist.bernoulli, p)

    optim = Adam({"lr": 0.1})
    inference = SVI(model, model, optim, loss="ELBO")
    inference.step()


@pytest.mark.xfail
def test_nested_iarange_bad():

    def model():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange-1", 100, 10):
            with pyro.iarange("iarange-2", 100, 10):
                pyro.sample("x", dist.bernoulli, p)

    optim = Adam({"lr": 0.1})
    inference = SVI(model, model, optim, loss="ELBO")
    with pytest.raises(SyntaxError):
        inference.step()
