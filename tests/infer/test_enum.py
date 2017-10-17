import itertools

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.optim
from pyro.infer import SVI
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.trace_elbo import Trace_ELBO
from tests.common import assert_equal, xfail_if_not_implemented

segfaults_on_old_pytorch = pytest.mark.skipif(
        torch.__version__ == '0.2.0_4',
        reason="pytorch segfaults at 0.2.0_4, fixed by 0.2.0+f964105")


# A purely discrete model, no batching.
def model0():
    p = pyro.param("p", Variable(torch.Tensor([0.05])))
    ps = pyro.param("ps", Variable(torch.Tensor([0.1, 0.2, 0.3, 0.4])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.Categorical(ps, one_hot=False))
    return dict(x=x, y=y)


# A discrete model with batching.
def model1():
    p = pyro.param("p", Variable(torch.Tensor([[0.05], [0.15]])))
    ps = pyro.param("ps", Variable(torch.Tensor([[0.1, 0.2, 0.3, 0.4],
                                                 [0.4, 0.3, 0.2, 0.1]])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.Categorical(ps, one_hot=False))
    assert x.size() == (2, 1)
    assert y.size() == (2, 1)
    return dict(x=x, y=y)


@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_scalar(graph_type):
    pyro.clear_param_store()
    traces = list(iter_discrete_traces(graph_type, model0))

    p = pyro.param("p").data
    ps = pyro.param("ps").data
    assert len(traces) == 2 * len(ps)

    for scale, trace in traces:
        x = trace.nodes["x"]["value"].data.long().view(-1)[0]
        y = trace.nodes["y"]["value"].data.long().view(-1)[0]
        expected_scale = Variable(torch.Tensor([[[1 - p[0], p[0]][x] * ps[y]]]))
        assert_equal(scale, expected_scale)


@pytest.mark.xfail(reason="https://github.com/uber/pyro/issues/220")
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_vector(graph_type):
    pyro.clear_param_store()
    traces = list(iter_discrete_traces(graph_type, model1))

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


# A simple Gaussian mixture model, with no vectorization.
def gmm_model(data, verbose=False):
    p = pyro.param("p", Variable(torch.Tensor([0.4]), requires_grad=True))
    sigma = pyro.param("sigma", Variable(torch.Tensor([1.0]), requires_grad=True))
    mus = pyro.param("mus", Variable(torch.Tensor([-1, 1]), requires_grad=True))
    for i in pyro.irange("data", len(data)):
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.size() == (1, 1)
        z = z.long().data[0, 0]
        if verbose:
            print("M{} z_{} = {}".format("  " * i, i, z))
        pyro.observe("x_{}".format(i), dist.DiagNormal(mus[z], sigma), data[i])


def gmm_guide(data, verbose=False):
    for i in pyro.irange("data", len(data)):
        p = pyro.param("p_{}".format(i), Variable(torch.Tensor([0.5]), requires_grad=True))
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.size() == (1, 1)
        z = z.long().data[0, 0]
        if verbose:
            print("G{} z_{} = {}".format("  " * i, i, z))


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_model, gmm_guide])
def test_gmm_iter_discrete_traces(model, data_size, graph_type):
    pyro.clear_param_store()
    data = Variable(torch.arange(0, data_size))
    traces = list(iter_discrete_traces(graph_type, model, data=data, verbose=True))
    # This non-vectorized version is exponential in data_size:
    assert len(traces) == 2 ** data_size


# A Gaussian mixture model, with vectorized batching.
def gmm_batch_model(data):
    p = pyro.param("p", Variable(torch.Tensor([0.4]), requires_grad=True))
    p = torch.cat([p, 1 - p])
    sigma = pyro.param("sigma", Variable(torch.ones(1), requires_grad=True))
    mus = pyro.param("mus", Variable(torch.Tensor([-1, 1]), requires_grad=True))
    with pyro.iarange("data", len(data)) as batch:
        n = len(batch)
        z = pyro.sample("z", dist.Categorical(p.unsqueeze(0).expand(n, 2)))
        assert z.size() == (n, 2)
        mu = torch.mv(z, mus)
        pyro.observe("x", dist.DiagNormal(mu, sigma.expand(n)), data[batch])


def gmm_batch_guide(data):
    with pyro.iarange("data", len(data)) as batch:
        n = len(batch)
        ps = pyro.param("ps", Variable(torch.ones(n, 1) * 0.5, requires_grad=True))
        ps = torch.cat([ps, 1 - ps], dim=1)
        z = pyro.sample("z", dist.Categorical(ps))
        assert z.size() == (n, 2)


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_batch_model, gmm_batch_guide])
def test_gmm_batch_iter_discrete_traces(model, data_size, graph_type):
    pyro.clear_param_store()
    data = Variable(torch.arange(0, data_size))
    traces = list(iter_discrete_traces(graph_type, model, data=data))
    # This vectorized version is independent of data_size:
    assert len(traces) == 2


@pytest.mark.parametrize("trace_graph", [False, True], ids=['dense', 'flat'])
@pytest.mark.parametrize("model,guide", [
    (gmm_model, gmm_guide),
    (gmm_batch_model, gmm_batch_guide),
], ids=['single', 'batch'])
@pytest.mark.parametrize("enum_discrete", [
    False,
    pytest.param(True, marks=segfaults_on_old_pytorch),
], ids=['sample', 'sum'])
def test_svi_step_smoke(model, guide, enum_discrete, trace_graph):
    pyro.clear_param_store()
    data = Variable(torch.Tensor([0, 1, 9]))

    optimizer = pyro.optim.Adam({"lr": .001})
    inference = SVI(model, guide, optimizer, loss="ELBO",
                    trace_graph=trace_graph, enum_discrete=enum_discrete)
    with xfail_if_not_implemented():
        inference.step(data)


def finite_difference(elbo, model, guide, data, delta=0.1):
    """
    Computes finite-difference approximation of all parameters.
    """
    params = pyro.get_param_store().get_all_param_names()
    assert params, "no params found"
    grads = {name: torch.zeros_like(pyro.param(name)) for name in params}
    for name in params:
        value = pyro.param(name).data
        for index in itertools.product(*map(range, value.size())):
            center = value[index]
            value[index] = center + delta
            pos = elbo.loss(model, guide, data)
            value[index] = center - delta
            neg = elbo.loss(model, guide, data)
            value[index] = center
            grads[name][index] = (pos - neg) / (2 * delta)
    return grads


@pytest.mark.parametrize("model,guide", [
    (gmm_model, gmm_guide),
    (gmm_batch_model, gmm_batch_guide),
], ids=['single', 'batch'])
@pytest.mark.parametrize("enum_discrete,num_particles", [
    pytest.param(True, 1, marks=pytest.mark.xfail),
    (False, 1000),
], ids=['sum', 'sample'])
def test_trace_elbo_gradient(model, guide, enum_discrete, num_particles):
    pyro.clear_param_store()
    data = Variable(torch.Tensor([0, 1]))

    print('Computing gradients using surrogate loss')
    elbo = Trace_ELBO(num_particles=num_particles, enum_discrete=enum_discrete)
    elbo.loss_and_grads(model, guide, data)
    params = pyro.get_param_store().get_all_param_names()
    assert params, "no params found"
    actual_grads = {name: pyro.param(name).grad.clone() for name in params}

    print('Computing gradients using finite difference')
    elbo = Trace_ELBO(num_particles=1000, enum_discrete=False)
    expected_grads = finite_difference(elbo, model, guide, data)

    for name in params:
        print('{} {}{}{}'.format(name, '-' * 30, actual_grads[name].data,
                                 expected_grads[name].data))
    assert_equal(actual_grads, expected_grads, prec=0.5)
