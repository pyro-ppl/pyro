from __future__ import absolute_import, division, print_function

import logging
import warnings

import pytest
import torch
from torch.autograd import Variable, variable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, config_enumerate
from pyro.optim import Adam

logger = logging.getLogger(__name__)

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
            logger.info(warning)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_nonempty_model_empty_guide_ok(trace_graph, enum_discrete):

    def model():
        mu = Variable(torch.Tensor([0, 0]))
        sigma = Variable(torch.Tensor([1, 1]))
        pyro.sample("x", dist.Normal(mu, sigma).reshape(extra_event_dims=1), obs=mu)

    def guide():
        pass

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_empty_model_empty_guide_ok(trace_graph, enum_discrete):

    def model():
        pass

    def guide():
        pass

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_variable_clash_in_model_error(trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("x", dist.Bernoulli(p))  # Should error here.

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    assert_error(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_model_guide_dim_mismatch_error(trace_graph, enum_discrete):

    def model():
        mu = Variable(torch.zeros(2))
        sigma = Variable(torch.zeros(2))
        pyro.sample("x", dist.Normal(mu, sigma))

    def guide():
        mu = pyro.param("mu", Variable(torch.zeros(2, 1), requires_grad=True))
        sigma = pyro.param("sigma", Variable(torch.zeros(2, 1), requires_grad=True))
        pyro.sample("x", dist.Normal(mu, sigma))

    assert_error(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_model_guide_shape_mismatch_error(trace_graph, enum_discrete):

    def model():
        mu = Variable(torch.zeros(1, 2))
        sigma = Variable(torch.zeros(1, 2))
        pyro.sample("x", dist.Normal(mu, sigma))

    def guide():
        mu = pyro.param("mu", Variable(torch.zeros(2, 1), requires_grad=True))
        sigma = pyro.param("sigma", Variable(torch.zeros(2, 1), requires_grad=True))
        pyro.sample("x", dist.Normal(mu, sigma))

    assert_error(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_variable_clash_in_guide_error(trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("x", dist.Bernoulli(p))  # Should error here.

    assert_error(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_irange_ok(subsample_size, trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        for i in pyro.irange("irange", 10, subsample_size):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for i in pyro.irange("irange", 10, subsample_size):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_irange_variable_clash_error(trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        for i in pyro.irange("irange", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for i in pyro.irange("irange", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.Bernoulli(p))

    assert_error(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_iarange_ok(subsample_size, trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        with pyro.iarange("iarange", 10, subsample_size) as ind:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        with pyro.iarange("iarange", 10, subsample_size) as ind:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_iarange_no_size_ok(trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        with pyro.iarange("iarange"):
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[10]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        with pyro.iarange("iarange"):
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[10]))

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_irange_irange_ok(subsample_size, trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        for i in pyro.irange("irange_0", 10, subsample_size):
            for j in pyro.irange("irange_1", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for i in pyro.irange("irange_0", 10, subsample_size):
            for j in pyro.irange("irange_1", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_irange_irange_swap_error(subsample_size, trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        for i in pyro.irange("irange_0", 10, subsample_size):
            for j in pyro.irange("irange_1", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for j in pyro.irange("irange_1", 10, subsample_size):
            for i in pyro.irange("irange_0", 10, subsample_size):
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    assert_error(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_irange_in_model_not_guide_ok(subsample_size, trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        for i in pyro.irange("irange", 10, subsample_size):
            pass
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_irange_in_guide_not_model_error(subsample_size, trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for i in pyro.irange("irange", 10, subsample_size):
            pass
        pyro.sample("x", dist.Bernoulli(p))

    assert_error(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_iarange_broadcast_error(trace_graph, enum_discrete):

    def model():
        p = variable(0.5, requires_grad=True)
        with pyro.iarange("iarange", 10, 5):
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[1]))

    assert_error(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.xfail(reason="RaoBlackwellization checks in trace_poutine need to be fixed to catch this")
def test_iarange_irange_warning():

    def model():
        p = variable(0.5)
        with pyro.iarange("iarange", 10, 5) as ind:
            for i in pyro.irange("irange", 10, 5):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            for i in pyro.irange("irange", 10, 5):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    assert_warning(model, guide, trace_graph=True)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_irange_iarange_ok(trace_graph, enum_discrete):

    def model():
        p = variable(0.5)
        for i in pyro.irange("irange", 10, 5):
            with pyro.iarange("iarange", 10, 5) as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for i in pyro.irange("irange", 10, 5):
            with pyro.iarange("iarange", 10, 5) as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    assert_ok(model, guide, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_nested_iarange_iarange_ok(trace_graph, enum_discrete):

    def model():
        p = variable(0.5, requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer)]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).reshape(sample_shape=[len(ind_inner), len(ind_outer)]))

    assert_ok(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_iarange_reuse_ok(trace_graph, enum_discrete):

    def model():
        p = variable(0.5, requires_grad=True)
        iarange_outer = pyro.iarange("iarange_outer", 10, 5, dim=-1)
        iarange_inner = pyro.iarange("iarange_inner", 11, 6, dim=-2)
        with iarange_outer as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer)]))
        with iarange_inner as ind_inner:
            pyro.sample("y", dist.Bernoulli(p).reshape(sample_shape=[len(ind_inner), 1]))
        with iarange_outer as ind_outer, iarange_inner as ind_inner:
            pyro.sample("z", dist.Bernoulli(p).reshape(sample_shape=[len(ind_inner), len(ind_outer)]))

    assert_ok(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_nested_iarange_iarange_dim_error_1(trace_graph, enum_discrete):

    def model():
        p = Variable(torch.Tensor([0.5]), requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer)]))  # error here
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).reshape(sample_shape=[len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer), len(ind_inner)]))

    assert_error(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_nested_iarange_iarange_dim_error_2(trace_graph, enum_discrete):

    def model():
        p = Variable(torch.Tensor([0.5]), requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer), 1]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer)]))  # error here
                pyro.sample("z", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer), len(ind_inner)]))

    assert_error(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_nested_iarange_iarange_dim_error_3(trace_graph, enum_discrete):

    def model():
        p = Variable(torch.Tensor([0.5]), requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer), 1]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).reshape(sample_shape=[len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).reshape(sample_shape=[len(ind_inner), 1]))  # error here

    assert_error(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_nested_iarange_iarange_dim_error_4(trace_graph, enum_discrete):

    def model():
        p = Variable(torch.Tensor([0.5]), requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer), 1]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).reshape(sample_shape=[len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).reshape(sample_shape=[len(ind_outer), len(ind_outer)]))  # error here

    assert_error(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_nonnested_iarange_iarange_ok(trace_graph, enum_discrete):

    def model():
        p = variable(0.5, requires_grad=True)
        with pyro.iarange("iarange_0", 10, 5) as ind1:
            pyro.sample("x0", dist.Bernoulli(p).reshape(sample_shape=[len(ind1)]))
        with pyro.iarange("iarange_1", 11, 6) as ind2:
            pyro.sample("x1", dist.Bernoulli(p).reshape(sample_shape=[len(ind2)]))

    assert_ok(model, model, trace_graph=trace_graph, enum_discrete=enum_discrete)


def test_three_indep_iarange_at_different_depths_ok():
    """
      /\
     /\ ia
    ia ia
    """
    def model():
        p = variable(0.5)
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.irange("irange1", 2):
                    with pyro.iarange("iarange1", 10, 5) as ind:
                        pyro.sample("y_%d" % j, dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))
            elif i == 1:
                with pyro.iarange("iarange1", 10, 5) as ind:
                    pyro.sample("z", dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.irange("irange1", 2):
                    with pyro.iarange("iarange1", 10, 5) as ind:
                        pyro.sample("y_%d" % j, dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))
            elif i == 1:
                with pyro.iarange("iarange1", 10, 5) as ind:
                    pyro.sample("z", dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    assert_ok(model, guide, trace_graph=True)


def test_iarange_wrong_size_error():

    def model():
        p = variable(0.5)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[1 + len(ind)]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[1 + len(ind)]))

    assert_error(model, guide, trace_graph=True)


def test_enum_discrete_single_ok():

    def model():
        p = variable(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    assert_ok(model, config_enumerate(guide), enum_discrete=True)


def test_enum_discrete_single_single_ok():

    def model():
        p = variable(0.5)
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("y", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("y", dist.Bernoulli(p))

    assert_ok(model, config_enumerate(guide), enum_discrete=True)


def test_enum_discrete_irange_single_ok():

    def model():
        p = variable(0.5)
        for i in pyro.irange("irange", 10, 5):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        for i in pyro.irange("irange", 10, 5):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    assert_ok(model, config_enumerate(guide), enum_discrete=True)


def test_iarange_enum_discrete_batch_ok():

    def model():
        p = variable(0.5)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[len(ind)]))

    assert_ok(model, config_enumerate(guide), enum_discrete=True)


def test_iarange_enum_discrete_no_discrete_vars_ok():

    def model():
        mu = variable(0.0)
        sigma = variable(1.0)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Normal(mu, sigma).reshape(sample_shape=[len(ind)]))

    def guide():
        mu = pyro.param("mu", variable(1.0, requires_grad=True))
        sigma = pyro.param("sigma", variable(2.0, requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Normal(mu, sigma).reshape(sample_shape=[len(ind)]))

    assert_ok(model, config_enumerate(guide), enum_discrete=True)


def test_no_iarange_enum_discrete_batch_error():

    def model():
        p = variable(0.5)
        pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[5]))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p).reshape(sample_shape=[5]))

    assert_error(model, config_enumerate(guide), enum_discrete=True)


@pytest.mark.parametrize('max_iarange_nesting', [0, 1, 2])
def test_enum_discrete_parallel_ok(max_iarange_nesting):
    iarange_shape = torch.Size([1] * max_iarange_nesting)

    def model():
        p = variable(0.5)
        x = pyro.sample("x", dist.Bernoulli(p))
        assert x.shape == torch.Size([2]) + iarange_shape

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        x = pyro.sample("x", dist.Bernoulli(p))
        assert x.shape == torch.Size([2]) + iarange_shape

    assert_ok(model, config_enumerate(guide, "parallel"), enum_discrete=True,
              max_iarange_nesting=max_iarange_nesting)


@pytest.mark.parametrize('max_iarange_nesting', [0, 1, 2])
def test_enum_discrete_parallel_nested_ok(max_iarange_nesting):
    iarange_shape = torch.Size([1] * max_iarange_nesting)

    def model():
        p2 = Variable(torch.ones(2) / 2)
        p3 = Variable(torch.ones(3) / 3)
        x2 = pyro.sample("x2", dist.OneHotCategorical(p2))
        x3 = pyro.sample("x3", dist.OneHotCategorical(p3))
        assert x2.shape == torch.Size([2]) + iarange_shape + p2.shape
        assert x3.shape == torch.Size([3, 1]) + iarange_shape + p3.shape

    assert_ok(model, config_enumerate(model, "parallel"), enum_discrete=True,
              max_iarange_nesting=max_iarange_nesting)


def test_enum_discrete_parallel_iarange_ok():
    enum_discrete = "defined below"

    def model():
        p2 = Variable(torch.ones(2) / 2)
        p34 = Variable(torch.ones(3, 4) / 4)
        p536 = Variable(torch.ones(5, 3, 6) / 6)

        x2 = pyro.sample("x2", dist.Categorical(p2))
        with pyro.iarange("iarange", 3):
            x34 = pyro.sample("x34", dist.Categorical(p34))
            with pyro.iarange("iarange", 5):
                x536 = pyro.sample("x536", dist.Categorical(p536))

        if not enum_discrete:
            # All dimensions are iarange dimensions.
            assert x2.shape == torch.Size([])
            assert x34.shape == torch.Size([3])
            assert x536.shape == torch.Size([5, 3])
        else:
            # Meaning of dimensions:    [ enum dims | iarange dims ]
            assert x2.shape == torch.Size([        2, 1, 1])  # noqa: E201
            assert x34.shape == torch.Size([    4, 1, 1, 3])  # noqa: E201
            assert x536.shape == torch.Size([6, 1, 1, 5, 3])  # noqa: E201

    enum_discrete = False
    assert_ok(model, model, enum_discrete=True, max_iarange_nesting=2)

    enum_discrete = True
    assert_ok(model, config_enumerate(model, "parallel"), enum_discrete=True,
              max_iarange_nesting=2)
