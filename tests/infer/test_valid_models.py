# This file tests a variety of model,guide pairs with valid and invalid structure.

import pytest

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI
from tests.common import segfaults_on_pytorch_020


def assert_ok(model, guide, **kwargs):
    inference = SVI(model, guide, Adam({"lr": 1e-3}), "ELBO", **kwargs)
    inference.step()


def assert_error(model, guide, **kwargs):
    inference = SVI(model,  guide, Adam({"lr": 1e-3}), "ELBO", **kwargs)
    with pytest.raises((NotImplementedError, UserWarning, KeyError)):
        inference.step()


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_irange_ok(trace_graph, subsample_size):

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange", 10, subsample_size):
            pyro.sample("x_{}".format(i), dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange", 10, subsample_size):
            pyro.sample("x_{}".format(i), dist.bernoulli, p)

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_iarange_ok(trace_graph, subsample_size):

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("irange", 10, subsample_size) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("irange", 10, subsample_size) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_irange_irange_ok(trace_graph, subsample_size):

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange_0", 10, subsample_size):
            for j in pyro.irange("irange_1", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange_0", 10, subsample_size):
            for j in pyro.irange("irange_1", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.bernoulli, p)

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_irange_irange_swap_error(trace_graph, subsample_size):

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange_0", 10, subsample_size):
            for j in pyro.irange("irange_1", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for j in pyro.irange("irange_1", 10, subsample_size):
            for i in pyro.irange("irange_0", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.bernoulli, p)

    assert_error(model, guide, trace_graph=trace_graph)


@pytest.mark.xfail(reason="raises UserWarning or KeyError")
@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_irange_in_model_not_guide_ok(trace_graph, subsample_size):

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange", 10, subsample_size):
            pass
        pyro.sample("x", dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_irange_in_guide_not_model_error(trace_graph, subsample_size):

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange", 10, subsample_size):
            pass
        pyro.sample("x", dist.bernoulli, p)

    assert_error(model, guide, trace_graph=trace_graph)


@pytest.mark.xfail(reason="NotImplementedError is not raised")
def test_iarange_irange_error():

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("iarange", 10, 5) as ind:
            for i in pyro.irange("irange", 10, 5):
                pyro.sample("x_{}".format(i), dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            for i in pyro.irange("irange", 10, 5):
                pyro.sample("x_{}".format(i), dist.bernoulli, p, batch_size=len(ind))

    assert_error(model, guide, trace_graph=True)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_irange_iarange_ok(trace_graph):

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange", 10, 5):
            with pyro.iarange("iarange", 10, 5) as ind:
                pyro.sample("x_{}".format(i), dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange", 10, 5):
            with pyro.iarange("iarange", 10, 5) as ind:
                pyro.sample("x_{}".format(i), dist.bernoulli, p, batch_size=len(ind))

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.xfail(reason="error is not caught")
def test_iarange_iarange_error():

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

    assert_error(model, guide, trace_graph=True)


@pytest.mark.xfail(reason="error is not caught")
def test_iarange_wrong_size_error():

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=1 + len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=1 + len(ind))

    assert_error(model, guide, trace_graph=True)


@segfaults_on_pytorch_020
def test_enum_discrete_single_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)

    assert_ok(model, guide, enum_discrete=True)


@segfaults_on_pytorch_020
def test_enum_discrete_single_single_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)
        pyro.sample("y", dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)
        pyro.sample("y", dist.bernoulli, p)

    assert_ok(model, guide, enum_discrete=True)


@segfaults_on_pytorch_020
def test_enum_discrete_irange_single_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange", 10, 5):
            pyro.sample("x_{}".format(i), dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange", 10, 5):
            pyro.sample("x_{}".format(i), dist.bernoulli, p)

    assert_ok(model, guide, enum_discrete=True)


@segfaults_on_pytorch_020
@pytest.mark.xfail(reason="tensor shape mismatch in: elbo_particle += ...")
def test_iarange_enum_discrete_batch_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    assert_ok(model, guide, enum_discrete=True)


@segfaults_on_pytorch_020
@pytest.mark.xfail(reason="error is not caught")
def test_no_iarange_enum_discrete_batch_error():

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p, batch_size=5)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p, batch_size=5)

    assert_error(model, guide, enum_discrete=True)


@segfaults_on_pytorch_020
@pytest.mark.xfail(reason="tensor shape mismatch in: elbo_particle += ...")
def test_enum_discrete_global_local_ok():
    # TODO Simplify this test when test_iarange_enum_discrete_batch_ok passes:
    test_iarange_enum_discrete_batch_ok_passes = False

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)
        if test_iarange_enum_discrete_batch_ok_passes:
            with pyro.iarange("iarange", 10, 5) as ind:
                pyro.sample("y", dist.bernoulli, p, batch_size=len(ind))
        else:
            pyro.sample("y", dist.bernoulli, p, batch_size=5)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)
        if test_iarange_enum_discrete_batch_ok_passes:
            with pyro.iarange("iarange", 10, 5) as ind:
                pyro.sample("y", dist.bernoulli, p, batch_size=len(ind))
        else:
            pyro.sample("y", dist.bernoulli, p, batch_size=5)

    assert_ok(model, guide, enum_discrete=True)
