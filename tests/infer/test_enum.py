from __future__ import absolute_import, division, print_function

import logging
import math

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
from tests.common import assert_equal

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

    p = pyro.param("p")
    ps = pyro.param("ps")
    assert len(traces) == 2 * len(ps)

    for scale, trace in traces:
        x = trace.nodes["x"]["value"].long()
        y = trace.nodes["y"]["value"].long()
        expected_scale = [1 - p, p][x] * ps[y]
        assert_equal(scale, expected_scale)


@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_vector(graph_type):
    pyro.clear_param_store()

    @config_enumerate
    def model():
        p = pyro.param("p", Variable(torch.Tensor([0.05, 0.15])))
        ps = pyro.param("ps", Variable(torch.Tensor([[0.1, 0.2, 0.3, 0.4],
                                                     [0.4, 0.3, 0.2, 0.1]])))
        with pyro.iarange("iarange", 2):
            x = pyro.sample("x", dist.Bernoulli(p))
            y = pyro.sample("y", dist.Categorical(ps))
        assert x.size() == (2,)
        assert y.size() == (2,)
        return dict(x=x, y=y)

    traces = list(iter_discrete_traces(graph_type, 1, model))

    p = pyro.param("p")
    ps = pyro.param("ps")
    assert len(traces) == 2 * ps.size(-1)

    for scale, trace in traces:
        x = trace.nodes["x"]["value"]
        y = trace.nodes["y"]["value"]
        log_prob = dist.Bernoulli(p).log_prob(x).sum() + dist.Categorical(ps).log_prob(y).sum()
        expected_scale = torch.exp(log_prob)
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
    loss = elbo.loss(model, guide).item()
    assert not math.isnan(loss), loss
    loss = elbo.loss_and_grads(model, guide)
    assert not math.isnan(loss), loss


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
            logger.debug("M{} z_{} = {}".format("  " * i, i, z.numpy()))
        pyro.observe("x_{}".format(i), dist.Normal(mus[z], sigma), data[i])


def gmm_guide(data, verbose=False):
    for i in pyro.irange("data", len(data)):
        p = pyro.param("p_{}".format(i), Variable(torch.Tensor([0.6]), requires_grad=True))
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.shape[-1:] == (1,)
        z = z.long()
        if verbose:
            logger.debug("G{} z_{} = {}".format("  " * i, i, z.numpy()))


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
        "\nexpected = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual = {}".format(actual_grad.detach().cpu().numpy()),
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
        "\nexpected = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual = {}".format(actual_grad.detach().cpu().numpy()),
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
        "\nexpected = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual = {}".format(actual_grad.detach().cpu().numpy()),
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
            "\nexpected = {}".format(expected_grad.detach().cpu().numpy()),
            "\n  actual = {}".format(actual_grad.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("iarange_dim", [1, 2])
@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
def test_iarange_elbo_gradient(iarange_dim, enum_discrete):
    pyro.clear_param_store()
    if not enum_discrete:
        num_particles = 1000  # Monte Carlo sample
    else:
        num_particles = 1  # a single particle should be exact

    def model():
        pyro.sample("y", dist.Bernoulli(0.25))
        with pyro.iarange("iarange", iarange_dim):
            pyro.sample("z", dist.Bernoulli(0.25).reshape(sample_shape=[iarange_dim]))

    def guide():
        q = pyro.param("q", variable(0.5, requires_grad=True))
        pyro.sample("y", dist.Bernoulli(q))
        with pyro.iarange("iarange", iarange_dim):
            pyro.sample("z", dist.Bernoulli(q).reshape(sample_shape=[iarange_dim]))

    logger.info("Computing gradients using surrogate loss")
    elbo = Trace_ELBO(num_particles=num_particles, max_iarange_nesting=1)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enum_discrete))
    actual_grad = pyro.param('q').grad

    logger.info("Computing analytic gradients")
    q = variable(0.5, requires_grad=True)
    expected_grad = (1 + iarange_dim) * grad(kl_divergence(dist.Bernoulli(q), dist.Bernoulli(0.25)), [q])[0]

    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected = {}".format(expected_grad.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
@pytest.mark.parametrize("pi1", [0.33, 0.43])
@pytest.mark.parametrize("pi2", [0.55, 0.27])
def test_non_mean_field_bern_bern_elbo_gradient(enum_discrete, pi1, pi2):
    pyro.clear_param_store()
    if not enum_discrete:
        num_particles = 1000  # Monte Carlo sample
    else:
        num_particles = 1  # a single particle should be exact

    def model():
        y = pyro.sample("y", dist.Bernoulli(0.33))
        pyro.sample("z", dist.Bernoulli(0.55 * y + 0.10))

    def guide():
        q1 = pyro.param("q1", variable(pi1, requires_grad=True))
        q2 = pyro.param("q2", variable(pi2, requires_grad=True))
        y = pyro.sample("y", dist.Bernoulli(q1))
        pyro.sample("z", dist.Bernoulli(q2 * y + 0.10))

    logger.info("Computing gradients using surrogate loss")
    elbo = Trace_ELBO(num_particles=num_particles, max_iarange_nesting=0)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enum_discrete))
    actual_grad_q1 = pyro.param('q1').grad
    actual_grad_q2 = pyro.param('q2').grad

    logger.info("Computing analytic gradients")
    q1 = variable(pi1, requires_grad=True)
    q2 = variable(pi2, requires_grad=True)
    elbo = kl_divergence(dist.Bernoulli(q1), dist.Bernoulli(0.33))
    elbo = elbo + q1 * kl_divergence(dist.Bernoulli(q2 + 0.10), dist.Bernoulli(0.65))
    elbo = elbo + (1.0 - q1) * kl_divergence(dist.Bernoulli(0.10), dist.Bernoulli(0.10))
    expected_grad_q1, expected_grad_q2 = grad(elbo, [q1, q2])

    prec = 0.03 if enum_discrete is None else 0.001

    assert_equal(actual_grad_q1, expected_grad_q1, prec=prec, msg="{q1}".join([
        "\nexpected = {}".format(expected_grad_q1.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad_q1.data.cpu().numpy()),
    ]))
    assert_equal(actual_grad_q2, expected_grad_q2, prec=prec, msg="{q2}".join([
        "\nexpected = {}".format(expected_grad_q2.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad_q2.data.cpu().numpy()),
    ]))


@pytest.mark.xfail(reason="Expensive; suggestion: run during large refactors")
@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
@pytest.mark.parametrize("pi1", [0.33, 0.44])
@pytest.mark.parametrize("pi2", [0.55, 0.39])
@pytest.mark.parametrize("pi3", [0.22, 0.29])
def test_non_mean_field_bern_normal_elbo_gradient(enum_discrete, pi1, pi2, pi3, include_z=True):
    pyro.clear_param_store()
    if enum_discrete is None:
        num_particles = 3000
    else:
        num_particles = 1000

    def model():
        q3 = pyro.param("q3", variable(pi3, requires_grad=True))
        y = pyro.sample("y", dist.Bernoulli(q3))
        if include_z:
            pyro.sample("z", dist.Normal(0.55 * y + q3, 1.0))

    def guide():
        q1 = pyro.param("q1", variable(pi1, requires_grad=True))
        q2 = pyro.param("q2", variable(pi2, requires_grad=True))
        y = pyro.sample("y", dist.Bernoulli(q1), infer={"enumerate": enum_discrete})
        if include_z:
            pyro.sample("z", dist.Normal(q2 * y + 0.10, 1.0))

    logger.info("Computing gradients using surrogate loss")
    elbo = Trace_ELBO(num_particles=num_particles, max_iarange_nesting=0)
    elbo.loss_and_grads(model, guide)
    actual_grad_q1 = pyro.param('q1').grad
    if include_z:
        actual_grad_q2 = pyro.param('q2').grad
    actual_grad_q3 = pyro.param('q3').grad

    logger.info("Computing analytic gradients")
    q1 = variable(pi1, requires_grad=True)
    q2 = variable(pi2, requires_grad=True)
    q3 = variable(pi3, requires_grad=True)
    elbo = kl_divergence(dist.Bernoulli(q1), dist.Bernoulli(q3))
    if include_z:
        elbo = elbo + q1 * kl_divergence(dist.Normal(q2 + 0.10, 1.0), dist.Normal(q3 + 0.55, 1.0))
        elbo = elbo + (1.0 - q1) * kl_divergence(dist.Normal(0.10, 1.0), dist.Normal(q3, 1.0))
        expected_grad_q1, expected_grad_q2, expected_grad_q3 = grad(elbo, [q1, q2, q3])
    else:
        expected_grad_q1, expected_grad_q3 = grad(elbo, [q1, q3])

    prec = 0.04 if enum_discrete is None else 0.02

    assert_equal(actual_grad_q1, expected_grad_q1, prec=prec, msg="{q1}".join([
        "\nexpected = {}".format(expected_grad_q1.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad_q1.data.cpu().numpy()),
    ]))
    if include_z:
        assert_equal(actual_grad_q2, expected_grad_q2, prec=prec, msg="{q2}".join([
            "\nexpected = {}".format(expected_grad_q2.data.cpu().numpy()),
            "\n  actual = {}".format(actual_grad_q2.data.cpu().numpy()),
        ]))
    assert_equal(actual_grad_q3, expected_grad_q3, prec=prec, msg="{q3}".join([
        "\nexpected = {}".format(expected_grad_q3.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad_q3.data.cpu().numpy()),
    ]))


# this test uses the non-enumerated MC estimator as ground truth
@pytest.mark.xfail(reason="Expensive; suggestion: run during large refactors")
@pytest.mark.parametrize("pi1", [0.33, 0.41])
@pytest.mark.parametrize("pi2", [0.44, 0.17])
@pytest.mark.parametrize("pi3", [0.22, 0.29])
def test_non_mean_field_normal_bern_elbo_gradient(pi1, pi2, pi3):

    def model():
        q3 = pyro.param("q3", variable(pi3, requires_grad=True))
        q4 = pyro.param("q4", variable(0.5 * (pi1 + pi2), requires_grad=True))
        z = pyro.sample("z", dist.Normal(q3, 1.0))
        zz = torch.exp(z) / (1.0 + torch.exp(z))
        pyro.sample("y", dist.Bernoulli(q4 * zz))

    def guide():
        q1 = pyro.param("q1", variable(pi1, requires_grad=True))
        q2 = pyro.param("q2", variable(pi2, requires_grad=True))
        z = pyro.sample("z", dist.Normal(q2, 1.0))
        zz = torch.exp(z) / (1.0 + torch.exp(z))
        pyro.sample("y", dist.Bernoulli(q1 * zz))

    qs = ['q1', 'q2', 'q3', 'q4']
    results = {}

    for ed, num_particles in zip([None, 'parallel', 'sequential'], [30000, 20000, 20000]):
        pyro.clear_param_store()
        elbo = Trace_ELBO(num_particles=num_particles, max_iarange_nesting=0)
        elbo.loss_and_grads(model, config_enumerate(guide, default=ed))
        results[str(ed)] = {}
        for q in qs:
            results[str(ed)]['actual_grad_%s' % q] = pyro.param(q).grad.data.cpu().numpy()

    prec = 0.03
    for ed in ['parallel', 'sequential']:
        print('\n*** %s ***' % ed)
        for q in qs:
            print("[%s] actual: " % q, results[ed]['actual_grad_%s' % q])

            msg = "{}".format(q).join(["\nexpected (MC estimate) = {}".format(results['None']['actual_grad_%s' % q]),
                                      "\n  actual ({} estimate)= {}".format(ed, results[ed]['actual_grad_%s' % q])])
            assert_equal(results[ed]['actual_grad_%s' % q], results['None']['actual_grad_%s' % q],
                         prec=prec, msg=msg)


@pytest.mark.parametrize("outer_dim", [1, 2])
@pytest.mark.parametrize("inner_dim", [1, 3])
@pytest.mark.parametrize("enum_discrete", [None, "sequential", "parallel"])
def test_nested_iarange_elbo_gradient(outer_dim, inner_dim, enum_discrete):
    pyro.clear_param_store()
    if not enum_discrete:
        num_particles = 1000  # Monte Carlo sample
    else:
        num_particles = 1  # a single particle should be exact

    def model():
        pyro.sample("x", dist.Bernoulli(0.25))
        with pyro.iarange("outer", outer_dim):
            pyro.sample("y", dist.Bernoulli(0.25).reshape(sample_shape=[outer_dim]))
            with pyro.iarange("inner", inner_dim):
                pyro.sample("z", dist.Bernoulli(0.25).reshape(sample_shape=[inner_dim, 1]))

    def guide():
        q = pyro.param("q", variable(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(q))
        with pyro.iarange("outer", outer_dim):
            pyro.sample("y", dist.Bernoulli(1 - q).reshape(sample_shape=[outer_dim]))
            with pyro.iarange("inner", inner_dim):
                pyro.sample("z", dist.Bernoulli(q).reshape(sample_shape=[inner_dim, 1]))

    logger.info("Computing gradients using surrogate loss")
    elbo = Trace_ELBO(num_particles=num_particles, max_iarange_nesting=1)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enum_discrete))
    actual_grad = pyro.param('q').grad

    logger.info("Computing analytic gradients")
    q = variable(0.5, requires_grad=True)
    expected_grad = (1 - outer_dim + inner_dim) * grad(kl_divergence(dist.Bernoulli(q), dist.Bernoulli(0.25)), [q])[0]

    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected = {}".format(expected_grad.data.cpu().numpy()),
        "\n  actual = {}".format(actual_grad.data.cpu().numpy()),
    ]))
