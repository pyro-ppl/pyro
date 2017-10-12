import os
import sys

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.optim
from pyro import poutine
from pyro.infer import SVI
from pyro.infer.enum import iter_discrete_traces, scale_trace
from tests.common import assert_equal


# A model with continuous and discrete variables, no batching.
def model0():
    p = pyro.param("p", Variable(torch.Tensor([0.1, 0.9])))
    mu = pyro.param("mu", Variable(torch.Tensor([-1.0, 1.0])))
    sigma = pyro.param("sigma", Variable(torch.Tensor([2.0, 3.0])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.DiagNormal(mu, sigma))
    z = pyro.sample("z", dist.DiagNormal(y, sigma))
    return dict(x=x, y=y, z=z)


# A purely discrete model, no batching.
def model1():
    p = pyro.param("p", Variable(torch.Tensor([0.05])))
    ps = pyro.param("ps", Variable(torch.Tensor([0.1, 0.2, 0.3, 0.4])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.Categorical(ps, one_hot=False))
    return dict(x=x, y=y)


# A discrete model with batching.
def model2():
    p = pyro.param("p", Variable(torch.Tensor([[0.05], [0.15]])))
    ps = pyro.param("ps", Variable(torch.Tensor([[0.1, 0.2, 0.3, 0.4],
                                                 [0.4, 0.3, 0.2, 0.1]])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.Categorical(ps, one_hot=False))
    assert x.size() == (2, 1)
    assert y.size() == (2, 1)
    return dict(x=x, y=y)


@pytest.mark.parametrize("model", [
    model0,
    pytest.param(model1, marks=pytest.mark.xfail(sys.version_info >= (3, 0),
                                                 reason="spurrious failure, probably pytorch")),
    pytest.param(model2, marks=pytest.mark.xfail(reason="https://github.com/uber/pyro/issues/253")),
])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_scale_trace(graph_type, model):
    pyro.get_param_store().clear()
    scale = 1.234
    tr1 = poutine.trace(model, graph_type=graph_type).get_trace()
    tr2 = scale_trace(tr1, scale)

    assert tr1 is not tr2, "scale_trace() did not make a copy"
    assert set(tr1.nodes.keys()) == set(tr2.nodes.keys())
    for name, site1 in tr1.nodes.items():
        site2 = tr2.nodes[name]
        assert site1 is not site2, "Trace.copy() was too shallow"
        if "scale" in site1:
            assert_equal(site2["scale"], scale * site1["scale"], msg=(site1, site2))

    # These check that memoized values were cleared.
    assert_equal(tr2.log_pdf(), scale * tr1.log_pdf())
    assert_equal(tr2.batch_log_pdf(), scale * tr1.batch_log_pdf())


@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_scalar(graph_type):
    pyro.get_param_store().clear()
    traces = list(iter_discrete_traces(graph_type, model1))

    p = pyro.param("p").data
    ps = pyro.param("ps").data
    assert len(traces) == 2 * len(ps)

    for scale, trace in traces:
        x = trace.nodes["x"]["value"].data.long().view(-1)[0]
        y = trace.nodes["y"]["value"].data.long().view(-1)[0]
        expected_scale = [1 - p[0], p[0]][x] * ps[y]
        assert_equal(scale, expected_scale)


@pytest.mark.xfail(reason="https://github.com/uber/pyro/issues/220")
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_vector(graph_type):
    pyro.get_param_store().clear()
    traces = list(iter_discrete_traces(graph_type, model2))

    p = pyro.param("p").data
    ps = pyro.param("ps").data
    assert len(traces) == 2 * ps.size(-1)

    for scale, trace in traces:
        x = trace.nodes["x"]["value"].data.squeeze().long()[0]
        y = trace.nodes["y"]["value"].data.squeeze().long()[0]
        expected_scale = torch.exp(dist.Bernoulli(p).log_pdf(x) *
                                   dist.Categorical(ps, one_hot=False).log_pdf(y))
        expected_scale = expected_scale.data.view(-1)[0]
        assert_equal(scale, expected_scale)


# A simple Gaussian mixture model.
def gmm_model(data):
    # p = pyro.param("p", Variable(torch.Tensor([0.5]), requires_grad=True))
    p = Variable(torch.Tensor([0.5]))
    sigma = Variable(torch.Tensor([1.0]))
    mus = pyro.param("mus", Variable(torch.Tensor([-1, 1]), requires_grad=True))
    for i in pyro.irange("data", len(data)):
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.size() == (1, 1)
        z = z.long().data[0, 0]
        print("M{} z_{} = {}".format("  " * i, i, z))
        pyro.observe("x_{}".format(i), dist.DiagNormal(mus[z], sigma), data[i])


def gmm_guide(data):
    for i in pyro.irange("data", len(data)):
        p = pyro.param("p_{}".format(i), Variable(torch.Tensor([0.5]), requires_grad=True))
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.size() == (1, 1)
        z = z.long().data[0, 0]
        print("G{} z_{} = {}".format("  " * i, i, z))


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_model, gmm_guide])
def test_gmm_iter_discrete_traces(model, data_size, graph_type):
    pyro.get_param_store().clear()
    data = Variable(torch.arange(0, data_size))
    traces = list(iter_discrete_traces(graph_type, model, data=data))
    assert len(traces) == 2 ** data_size


@pytest.mark.parametrize("trace_graph", [False, True])
@pytest.mark.parametrize("enum_discrete", [
    False,
    pytest.param(True, marks=pytest.mark.xfail(run=False, reason="segfault")),
])
def test_gmm_elbo_single_datum(enum_discrete, trace_graph):
    pyro.get_param_store().clear()
    data = Variable(torch.Tensor([0, 1, 9]))

    optimizer = pyro.optim.Adam({"lr": .001})
    inference = SVI(gmm_model, gmm_guide, optimizer, loss="ELBO",
                    trace_graph=trace_graph, enum_discrete=enum_discrete)
    inference.step(data)


# This is useful for reproducing the segfaulting test.
if __name__ == '__main__':
    test_gmm_elbo_single_datum(enum_discrete=True, trace_graph=False)
