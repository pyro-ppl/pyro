# This file tests a variety of model,guide pairs with valid and invalid syntax.

import pytest

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI


def assert_ok(model, guide):
    inference = SVI(model, guide, Adam({"lr": 1e-3}), "ELBO", trace_graph=True)
    inference.step()


def assert_error(model, guide):
    inference = SVI(model,  guide, Adam({"lr": 1e-3}), "ELBO", trace_graph=True)
    with pytest.raises(SyntaxError):
        inference.step()


def test_irange_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange", 10, 5):
            pyro.sample("x_{}".format(i), dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange", 10, 5):
                pyro.sample("x_{}".format(i), dist.bernoulli, p)

    assert_ok(model, guide)


def test_iarange_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("irange", 10, 5) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("irange", 10, 5) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    assert_ok(model, guide)


@pytest.mark.xfail(reason="nested replay appears to be broken")
def test_nested_irange_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange_0", 10, 5):
            for j in pyro.irange("irange_1", 10, 5):
                pyro.sample("x_{}_{}".format(i, j), dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange_0", 10, 5):
            for j in pyro.irange("irange_1", 10, 5):
                pyro.sample("x_{}_{}".format(i, j), dist.bernoulli, p)

    assert_ok(model, guide)


@pytest.mark.xfail(reason="error is not caught")
def test_nested_iarange_error():

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("iarange_0", 10, 5) as ind1:
            with pyro.iarange("iarange_1", 10, 5) as ind2:
                pyro.sample("x", dist.bernoulli, p, batch_size=len(ind1) * len(ind2))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange_0", 10, 5) as ind1:
            with pyro.iarange("iarange_1", 10, 5) as ind2:
                pyro.sample("x", dist.bernoulli, p, batch_size=len(ind1) * len(ind2))

    assert_error(model, guide)
