from __future__ import absolute_import, division, print_function

import warnings

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam

# This file tests a variety of model,guide pairs with valid and invalid structure.


def assert_ok(model, guide, **kwargs):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.clear_param_store()
    inference = SVI(model, guide, Adam({"lr": 1e-6}), "ELBO", **kwargs)
    inference.step()


def assert_error(model, guide, **kwargs):
    """
    Assert that inference fails with an error.
    """
    pyro.clear_param_store()
    inference = SVI(model,  guide, Adam({"lr": 1e-6}), "ELBO", **kwargs)
    with pytest.raises((NotImplementedError, UserWarning, KeyError, ValueError, RuntimeError)):
        inference.step()


def assert_warning(model, guide, **kwargs):
    """
    Assert that inference works but with a warning.
    """
    pyro.clear_param_store()
    inference = SVI(model,  guide, Adam({"lr": 1e-6}), "ELBO", **kwargs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inference.step()
        assert len(w), 'No warnings were raised'
        for warning in w:
            print(warning)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_nonempty_model_empty_guide_ok(trace_graph):

    def model():
        mu = Variable(torch.Tensor([0, 0]))
        sigma = Variable(torch.Tensor([1, 1]))
        pyro.sample("x", dist.normal, mu, sigma, obs=mu)

    def guide():
        pass

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_empty_model_empty_guide_ok(trace_graph):

    def model():
        pass

    def guide():
        pass

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_variable_clash_in_model_error(trace_graph):

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)
        pyro.sample("x", dist.bernoulli, p)  # Should error here.

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)

    assert_error(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_model_guide_dim_mismatch_error(trace_graph):

    def model():
        mu = Variable(torch.zeros(2))
        sigma = Variable(torch.zeros(2))
        pyro.sample("x", dist.normal, mu, sigma)

    def guide():
        mu = pyro.param("mu", Variable(torch.zeros(2, 1), requires_grad=True))
        sigma = pyro.param("sigma", Variable(torch.zeros(2, 1), requires_grad=True))
        pyro.sample("x", dist.normal, mu, sigma)

    assert_error(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_model_guide_shape_mismatch_error(trace_graph):

    def model():
        mu = Variable(torch.zeros(1, 2))
        sigma = Variable(torch.zeros(1, 2))
        pyro.sample("x", dist.normal, mu, sigma)

    def guide():
        mu = pyro.param("mu", Variable(torch.zeros(2, 1), requires_grad=True))
        sigma = pyro.param("sigma", Variable(torch.zeros(2, 1), requires_grad=True))
        pyro.sample("x", dist.normal, mu, sigma)

    assert_error(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_variable_clash_in_guide_error(trace_graph):

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)
        pyro.sample("x", dist.bernoulli, p)  # Should error here.

    assert_error(model, guide, trace_graph=trace_graph)


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


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_irange_variable_clash_error(trace_graph):

    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.bernoulli, p)

    assert_error(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_iarange_ok(trace_graph, subsample_size):

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("iarange", 10, subsample_size) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange", 10, subsample_size) as ind:
            pyro.sample("x", dist.bernoulli, p, batch_size=len(ind))

    assert_ok(model, guide, trace_graph=trace_graph)


@pytest.mark.parametrize("trace_graph", [False, True], ids=["trace", "tracegraph"])
def test_iarange_no_size_ok(trace_graph):

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("iarange"):
            pyro.sample("x", dist.bernoulli, p, batch_size=10)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange"):
            pyro.sample("x", dist.bernoulli, p, batch_size=10)

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


def test_iarange_irange_warning():

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

    assert_warning(model, guide, trace_graph=True)


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


def test_nested_iarange_iarange_warning():

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

    assert_warning(model, guide, trace_graph=True)


def test_nonnested_iarange_iarange_warning():

    def model():
        p = Variable(torch.Tensor([0.5]))
        with pyro.iarange("iarange_0", 10, 5) as ind1:
            pyro.sample("x0", dist.bernoulli, p, batch_size=len(ind1))
        with pyro.iarange("iarange_1", 10, 5) as ind2:
            pyro.sample("x1", dist.bernoulli, p, batch_size=len(ind2))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        with pyro.iarange("iarange_0", 10, 5) as ind1:
            pyro.sample("x0", dist.bernoulli, p, batch_size=len(ind1))
        with pyro.iarange("iarange_1", 10, 5) as ind2:
            pyro.sample("x1", dist.bernoulli, p, batch_size=len(ind2))

    assert_warning(model, guide, trace_graph=True)


def test_three_indep_iarange_at_different_depths_ok():
    """
      /\
     /\ ia
    ia ia
    """
    def model():
        p = Variable(torch.Tensor([0.5]))
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.bernoulli, p)
            if i == 0:
                for j in pyro.irange("irange1", 2):
                    with pyro.iarange("iarange1", 10, 5) as ind:
                        pyro.sample("y_%d" % j, dist.bernoulli, p, batch_size=len(ind))
            elif i == 1:
                with pyro.iarange("iarange1", 10, 5) as ind:
                    pyro.sample("z", dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.bernoulli, p)
            if i == 0:
                for j in pyro.irange("irange1", 2):
                    with pyro.iarange("iarange1", 10, 5) as ind:
                        pyro.sample("y_%d" % j, dist.bernoulli, p, batch_size=len(ind))
            elif i == 1:
                with pyro.iarange("iarange1", 10, 5) as ind:
                    pyro.sample("z", dist.bernoulli, p, batch_size=len(ind))

    assert_ok(model, guide, trace_graph=True)


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


def test_enum_discrete_single_ok():

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)

    assert_ok(model, guide, enum_discrete=True)


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


def test_iarange_enum_discrete_no_discrete_vars_ok():

    def model():
        mu = Variable(torch.zeros(2, 1))
        sigma = Variable(torch.ones(2, 1))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.normal, mu, sigma, batch_size=len(ind))

    def guide():
        mu = pyro.param("mu", Variable(torch.zeros(2, 1), requires_grad=True))
        sigma = pyro.param("sigma", Variable(torch.ones(2, 1), requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.normal, mu, sigma, batch_size=len(ind))

    assert_ok(model, guide, enum_discrete=True)


def test_no_iarange_enum_discrete_batch_error():

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p, batch_size=5)

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p, batch_size=5)

    assert_error(model, guide, enum_discrete=True)


def test_enum_discrete_global_local_error():

    def model():
        p = Variable(torch.Tensor([0.5]))
        pyro.sample("x", dist.bernoulli, p)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("y", dist.bernoulli, p, batch_size=len(ind))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
        pyro.sample("x", dist.bernoulli, p)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("y", dist.bernoulli, p, batch_size=len(ind))

    assert_error(model, guide, enum_discrete=True)
