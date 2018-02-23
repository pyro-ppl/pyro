from __future__ import absolute_import, division, print_function

import itertools
import logging
import math
import os

import pytest
import torch
from torch.autograd import Variable, grad, variable
from torch.distributions import kl_divergence

import pyro
import pyro.distributions as dist
import pyro.optim
from pyro.infer import SVI, config_enumerate
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.tracegraph_elbo import TraceGraph_ELBO
from tests.common import assert_equal, xfail_if_not_implemented

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_scalar(graph_type):
    pyro.clear_param_store()

    @config_enumerate
    def model():
        p = pyro.param("p", variable(0.05))
        ps = pyro.param("ps", Variable(torch.Tensor([0.1, 0.2, 0.3, 0.4])))
        x = pyro.sample("x", dist.Bernoulli(p))
        y = pyro.sample("y", dist.Categorical(ps))
        return dict(x=x, y=y)

    traces = list(iter_discrete_traces(graph_type, 0, model))

    p = pyro.param("p").data
    ps = pyro.param("ps").data
    assert len(traces) == 2 * len(ps)

    for scale, trace in traces:
        x = trace.nodes["x"]["value"].data.long().view(-1)[0]
        y = trace.nodes["y"]["value"].data.long().view(-1)[0]
        expected_scale = Variable(torch.Tensor([[1 - p[0], p[0]][x] * ps[y]]))
        assert_equal(scale, expected_scale)


@pytest.mark.xfail(reason="https://github.com/uber/pyro/issues/220")
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_vector(graph_type):
    pyro.clear_param_store()

    @config_enumerate
    def model():
        p = pyro.param("p", Variable(torch.Tensor([[0.05], [0.15]])))
        ps = pyro.param("ps", Variable(torch.Tensor([[0.1, 0.2, 0.3, 0.4],
                                                     [0.4, 0.3, 0.2, 0.1]])))
        x = pyro.sample("x", dist.Bernoulli(p))
        y = pyro.sample("y", dist.Categorical(ps))
        assert x.size() == (2, 1)
        assert y.size() == (2, 1)
        return dict(x=x, y=y)

    traces = list(iter_discrete_traces(graph_type, 0, model))

    p = pyro.param("p").data
    ps = pyro.param("ps").data
    assert len(traces) == 2 * ps.size(-1)

    for scale, trace in traces:
        x = trace.nodes["x"]["value"].data.squeeze().long()[0]
        y = trace.nodes["y"]["value"].data.squeeze().long()[0]
        expected_scale = torch.exp(dist.Bernoulli(p).log_pdf(x) * dist.Categorical(ps).log_pdf(y))
        expected_scale = expected_scale.data.view(-1)[0]
        assert_equal(scale, expected_scale)


@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["dense", "flat"])
def test_iter_discrete_traces_nan(enum_discrete, trace_graph):
    pyro.clear_param_store()

    def model():
        p = Variable(torch.Tensor([0.0, 0.5, 1.0]))
        pyro.sample("z", dist.Bernoulli(p).reshape(extra_event_dims=1))

    def guide():
        p = pyro.param("p", Variable(torch.Tensor([0.0, 0.5, 1.0]), requires_grad=True))
        pyro.sample("z", dist.Bernoulli(p).reshape(extra_event_dims=1))

    guide = config_enumerate(guide, default=enum_discrete)
    Elbo = TraceGraph_ELBO if trace_graph else Trace_ELBO
    elbo = Elbo(max_iarange_nesting=0)
    with xfail_if_not_implemented():
        loss = elbo.loss(model, guide)
        assert isinstance(loss, float) and not math.isnan(loss), loss
        loss = elbo.loss_and_grads(model, guide)
        assert isinstance(loss, float) and not math.isnan(loss), loss


# A simple Gaussian mixture model, with no vectorization.
def gmm_model(data, verbose=False):
    p = pyro.param("p", Variable(torch.Tensor([0.3]), requires_grad=True))
    sigma = pyro.param("sigma", Variable(torch.Tensor([1.0]), requires_grad=True))
    mus = Variable(torch.Tensor([-1, 1]))
    for i in pyro.irange("data", len(data)):
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.shape[-1:] == (1,)
        z = z.long()
        if verbose:
            logger.debug("M{} z_{} = {}".format("  " * i, i, z))
        pyro.observe("x_{}".format(i), dist.Normal(mus[z], sigma), data[i])


def gmm_guide(data, verbose=False):
    for i in pyro.irange("data", len(data)):
        p = pyro.param("p_{}".format(i), Variable(torch.Tensor([0.6]), requires_grad=True))
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.shape[-1:] == (1,)
        z = z.long()
        if verbose:
            logger.debug("G{} z_{} = {}".format("  " * i, i, z))


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_model, gmm_guide])
def test_gmm_iter_discrete_traces(model, data_size, graph_type):
    pyro.clear_param_store()
    data = Variable(torch.arange(0, data_size))
    model = config_enumerate(model)
    traces = list(iter_discrete_traces(graph_type, 0, model, data=data, verbose=True))
    # This non-vectorized version is exponential in data_size:
    assert len(traces) == 2**data_size


# A Gaussian mixture model, with vectorized batching.
def gmm_batch_model(data):
    p = pyro.param("p", Variable(torch.Tensor([0.3]), requires_grad=True))
    p = torch.cat([p, 1 - p])
    sigma = pyro.param("sigma", Variable(torch.Tensor([1.0]), requires_grad=True))
    mus = Variable(torch.Tensor([-1, 1]))
    with pyro.iarange("data", len(data)) as batch:
        n = len(batch)
        z = pyro.sample("z", dist.OneHotCategorical(p).reshape(sample_shape=[n]))
        assert z.shape[-2:] == (n, 2)
        mu = (z * mus).sum(-1)
        pyro.observe("x", dist.Normal(mu, sigma.expand(n)), data[batch])


def gmm_batch_guide(data):
    with pyro.iarange("data", len(data)) as batch:
        n = len(batch)
        ps = pyro.param("ps", Variable(torch.ones(n, 1) * 0.6, requires_grad=True))
        ps = torch.cat([ps, 1 - ps], dim=1)
        z = pyro.sample("z", dist.OneHotCategorical(ps))
        assert z.shape[-2:] == (n, 2)


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_batch_model, gmm_batch_guide])
def test_gmm_batch_iter_discrete_traces(model, data_size, graph_type):
    pyro.clear_param_store()
    data = Variable(torch.arange(0, data_size))
    model = config_enumerate(model)
    traces = list(iter_discrete_traces(graph_type, 1, model, data=data))
    # This vectorized version is independent of data_size:
    assert len(traces) == 2


@pytest.mark.parametrize("trace_graph", [False, True], ids=["dense", "flat"])
@pytest.mark.parametrize("model,guide", [
    (gmm_model, gmm_guide),
    (gmm_batch_model, gmm_batch_guide),
], ids=["single", "batch"])
@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
def test_svi_step_smoke(model, guide, enum_discrete, trace_graph):
    pyro.clear_param_store()
    data = Variable(torch.Tensor([0, 1, 9]))

    guide = config_enumerate(guide, default=enum_discrete)
    optimizer = pyro.optim.Adam({"lr": .001})
    inference = SVI(model, guide, optimizer, loss="ELBO",
                    trace_graph=trace_graph, max_iarange_nesting=1)
    with xfail_if_not_implemented():
        inference.step(data)


@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["dense", "flat"])
def test_bern_elbo_gradient(enum_discrete, trace_graph):
    pyro.clear_param_store()
    if not enum_discrete:
        num_particles = 100  # Monte Carlo sample
    elif trace_graph:
        num_particles = 100  # TraceGraph_ELBO silently ignores enumration
    else:
        num_particles = 1  # a single particle should be exact

    def model():
        pyro.sample("z", dist.Bernoulli(0.25))

    def guide():
        q = pyro.param("q", variable(0.5, requires_grad=True))
        pyro.sample("z", dist.Bernoulli(q))

    logger.info("Computing gradients using surrogate loss")
    Elbo = TraceGraph_ELBO if trace_graph else Trace_ELBO
    elbo = Elbo(num_particles=num_particles, max_iarange_nesting=0)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enum_discrete))
    actual_grad = pyro.param('q').grad

    logger.info("Computing analytic gradients")
    q = variable(0.5, requires_grad=True)
    expected_grad = grad(kl_divergence(dist.Bernoulli(q), dist.Bernoulli(0.25)), [q])[0]

    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected = {}".format(expected_grad.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["dense", "flat"])
def test_bern_bern_elbo_gradient(enum_discrete, trace_graph):
    pyro.clear_param_store()
    if not enum_discrete:
        num_particles = 500  # Monte Carlo sample
    elif trace_graph:
        num_particles = 500  # TraceGraph_ELBO silently ignores enumeration
    else:
        num_particles = 1  # a single particle should be exact

    def model():
        pyro.sample("y", dist.Bernoulli(0.25))
        pyro.sample("z", dist.Bernoulli(0.25))

    def guide():
        q = pyro.param("q", variable(0.5, requires_grad=True))
        pyro.sample("z", dist.Bernoulli(q))
        pyro.sample("y", dist.Bernoulli(q))

    logger.info("Computing gradients using surrogate loss")
    Elbo = TraceGraph_ELBO if trace_graph else Trace_ELBO
    elbo = Elbo(num_particles=num_particles, max_iarange_nesting=0)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enum_discrete))
    actual_grad = pyro.param('q').grad

    logger.info("Computing analytic gradients")
    q = variable(0.5, requires_grad=True)
    expected_grad = 2 * grad(kl_divergence(dist.Bernoulli(q), dist.Bernoulli(0.25)), [q])[0]

    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected = {}".format(expected_grad.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate1", ["sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", ["sequential", "parallel", None])
@pytest.mark.parametrize("enumerate3", ["sequential", "parallel"])
def test_berns_elbo_gradient(enumerate1, enumerate2, enumerate3):
    pyro.clear_param_store()
    if all([enumerate1, enumerate2, enumerate3]):
        num_particles = 1
        prec = 0.001
    else:
        num_particles = 1000
        prec = 0.1

    def model():
        pyro.sample("x1", dist.Bernoulli(0.1))
        pyro.sample("x2", dist.Bernoulli(0.2))
        pyro.sample("x3", dist.Bernoulli(0.3))

    def guide():
        p = pyro.param("p", variable(0.5, requires_grad=True))
        pyro.sample("x1", dist.Bernoulli(p), infer={"enumerate": enumerate1})
        pyro.sample("x2", dist.Bernoulli(p), infer={"enumerate": enumerate2})
        pyro.sample("x3", dist.Bernoulli(p), infer={"enumerate": enumerate3})

    logger.info("Computing gradients using surrogate loss")
    elbo = Trace_ELBO(num_particles=num_particles, max_iarange_nesting=0)
    elbo.loss_and_grads(model, guide)
    actual_grad = pyro.param('p').grad

    logger.info("Computing analytic gradients")
    p = variable(0.5, requires_grad=True)
    kl = sum(kl_divergence(dist.Bernoulli(p), dist.Bernoulli(p0)) for p0 in [0.1, 0.2, 0.3])
    expected_grad = grad(kl, [p])[0]

    assert_equal(actual_grad, expected_grad, prec=prec, msg="".join([
        "\nexpected = {}".format(expected_grad.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate1", ["sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", ["sequential", "parallel"])
@pytest.mark.parametrize("enumerate3", ["sequential", "parallel"])
@pytest.mark.parametrize("max_iarange_nesting", [0, 1])
def test_categoricals_elbo_gradient(enumerate1, enumerate2, enumerate3, max_iarange_nesting):
    pyro.clear_param_store()
    p1 = variable([0.6, 0.4])
    p2 = variable([0.3, 0.3, 0.4])
    p3 = variable([0.1, 0.2, 0.3, 0.4])
    q1 = pyro.param("q1", variable([0.4, 0.6], requires_grad=True))
    q2 = pyro.param("q2", variable([0.4, 0.3, 0.3], requires_grad=True))
    q3 = pyro.param("q3", variable([0.4, 0.3, 0.2, 0.1], requires_grad=True))

    def model():
        pyro.sample("x1", dist.Categorical(p1))
        pyro.sample("x2", dist.Categorical(p2))
        pyro.sample("x3", dist.Categorical(p3))

    def guide():
        pyro.sample("x1", dist.Categorical(pyro.param("q1")), infer={"enumerate": enumerate1})
        pyro.sample("x2", dist.Categorical(pyro.param("q2")), infer={"enumerate": enumerate2})
        pyro.sample("x3", dist.Categorical(pyro.param("q3")), infer={"enumerate": enumerate3})

    logger.info("Computing analytic gradients")
    kl = (kl_divergence(dist.Categorical(q1), dist.Categorical(p1)) +
          kl_divergence(dist.Categorical(q2), dist.Categorical(p2)) +
          kl_divergence(dist.Categorical(q3), dist.Categorical(p3)))
    expected_grads = grad(kl, [q1, q2, q3], create_graph=True)

    logger.info("Computing gradients using surrogate loss")
    elbo = Trace_ELBO(max_iarange_nesting=max_iarange_nesting)
    elbo.loss_and_grads(model, guide)
    actual_grads = [q1.grad, q2.grad, q3.grad]

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_equal(actual_grad, expected_grad, prec=0.001, msg="".join([
            "\nexpected = {}".format(expected_grad.data.cpu().numpy()),
            "\n  actual = {}".format(actual_grad.data.cpu().numpy()),
        ]))


def finite_difference(eval_loss, delta=0.1):
    """
    Computes finite-difference approximation of all parameters.
    """
    params = pyro.get_param_store().get_all_param_names()
    assert params, "no params found"
    grads = {name: Variable(torch.zeros(pyro.param(name).size())) for name in params}
    for name in sorted(params):
        value = pyro.param(name).data
        for index in itertools.product(*map(range, value.size())):
            center = value[index]
            value[index] = center + delta
            pos = eval_loss()
            value[index] = center - delta
            neg = eval_loss()
            value[index] = center
            grads[name][index] = (pos - neg) / (2 * delta)
    return grads


@pytest.mark.parametrize("model,guide", [
    (gmm_model, gmm_guide),
    (gmm_batch_model, gmm_batch_guide),
], ids=["single", "batch"])
@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["dense", "flat"])
def test_gmm_elbo_gradient(model, guide, enum_discrete, trace_graph):
    pyro.clear_param_store()
    data = Variable(torch.Tensor([-1, 1]))
    diff_particles = 4000
    if not enum_discrete:
        elbo_particles = 1000  # Monte Carlo sample
    elif trace_graph:
        elbo_particles = 1000  # TraceGraph_ELBO silently ignores enumeration
    else:
        elbo_particles = 1  # a single particle should be exact

    if elbo_particles > 1 and 'CI' in os.environ:
        pytest.skip(reason='slow test')

    logger.info("Computing gradients using surrogate loss")
    Elbo = TraceGraph_ELBO if trace_graph else Trace_ELBO
    elbo = Elbo(num_particles=elbo_particles, max_iarange_nesting=1)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enum_discrete), data)
    params = sorted(pyro.get_param_store().get_all_param_names())
    assert params, "no params found"
    actual_grads = {name: pyro.param(name).grad.clone() for name in params}

    logger.info("Computing gradients using finite difference")
    elbo = Trace_ELBO(num_particles=diff_particles, max_iarange_nesting=1)
    expected_grads = finite_difference(lambda: elbo.loss(model, guide, data))

    for name in params:
        logger.info("\n".join([
            "{} {}".format(name, "-" * 30),
            "expected = {}".format(expected_grads[name].data.cpu().numpy()),
            "  actual = {}".format(actual_grads[name].data.cpu().numpy()),
        ]))
    assert_equal(actual_grads, expected_grads, prec=0.5)
