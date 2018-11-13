from __future__ import absolute_import, division, print_function

import warnings
import logging

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints, kl_divergence

import pyro
import pyro.distributions as dist
import pyro.ops.jit
import pyro.poutine as poutine
from pyro.infer import (SVI, JitTrace_ELBO, JitTraceEnum_ELBO, JitTraceGraph_ELBO, Trace_ELBO, TraceEnum_ELBO,
                        TraceGraph_ELBO)
from pyro.optim import Adam
from pyro.poutine.indep_messenger import CondIndepStackFrame
from tests.common import assert_equal, xfail_param


def constant(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        return torch.tensor(*args, **kwargs)


logger = logging.getLogger(__name__)


def test_simple():
    y = torch.ones(2)

    def f(x):
        logger.debug('Inside f')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            assert x is y
        return y + 1.0

    logger.debug('Compiling f')
    f = torch.jit.trace(f, (y,), check_trace=False)
    logger.debug('Calling f(y)')
    assert_equal(f(y), y.new_tensor([2., 2.]))
    logger.debug('Calling f(y)')
    assert_equal(f(y), y.new_tensor([2., 2.]))
    logger.debug('Calling f(torch.zeros(2))')
    assert_equal(f(torch.zeros(2)), y.new_tensor([1., 1.]))
    logger.debug('Calling f(torch.zeros(5))')
    assert_equal(f(torch.ones(5)), y.new_tensor([2., 2., 2., 2., 2.]))


def test_multi_output():
    y = torch.ones(2)

    def f(x):
        logger.debug('Inside f')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            assert x is y
        return y - 1.0, y + 1.0

    logger.debug('Compiling f')
    f = torch.jit.trace(f, (y,), check_trace=False)
    logger.debug('Calling f(y)')
    assert_equal(f(y)[1], y.new_tensor([2., 2.]))
    logger.debug('Calling f(y)')
    assert_equal(f(y)[1], y.new_tensor([2., 2.]))
    logger.debug('Calling f(torch.zeros(2))')
    assert_equal(f(torch.zeros(2))[1], y.new_tensor([1., 1.]))
    logger.debug('Calling f(torch.zeros(5))')
    assert_equal(f(torch.ones(5))[1], y.new_tensor([2., 2., 2., 2., 2.]))


def test_backward():
    y = torch.ones(2, requires_grad=True)

    def f(x):
        logger.debug('Inside f')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            assert x is y
        return (y + 1.0).sum()

    logger.debug('Compiling f')
    f = torch.jit.trace(f, (y,), check_trace=False)
    logger.debug('Calling f(y)')
    f(y).backward()
    logger.debug('Calling f(y)')
    f(y)
    logger.debug('Calling f(torch.zeros(2))')
    f(torch.zeros(2, requires_grad=True))
    logger.debug('Calling f(torch.zeros(5))')
    f(torch.ones(5, requires_grad=True))


@pytest.mark.xfail(reason="grad cannot appear in jitted code")
def test_grad():

    def f(x, y):
        logger.debug('Inside f')
        loss = (x - y).pow(2).sum()
        return torch.autograd.grad(loss, [x, y], allow_unused=True)

    logger.debug('Compiling f')
    f = torch.jit.trace(f, (torch.zeros(2, requires_grad=True), torch.ones(2, requires_grad=True)))
    logger.debug('Invoking f')
    f(torch.zeros(2, requires_grad=True), torch.ones(2, requires_grad=True))
    logger.debug('Invoking f')
    f(torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True))


@pytest.mark.xfail(reason="grad cannot appear in jitted code")
def test_grad_expand():

    def f(x, y):
        logger.debug('Inside f')
        loss = (x - y).pow(2).sum()
        return torch.autograd.grad(loss, [x, y], allow_unused=True)

    logger.debug('Compiling f')
    f = torch.jit.trace(f, (torch.zeros(2, requires_grad=True), torch.ones(1, requires_grad=True)))
    logger.debug('Invoking f')
    f(torch.zeros(2, requires_grad=True), torch.ones(1, requires_grad=True))
    logger.debug('Invoking f')
    f(torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True))


@pytest.mark.xfail(reason="https://github.com/pytorch/pytorch/issues/11555")
def test_masked_fill():

    def f(y, mask):
        return y.clone().masked_fill_(mask, 0.)

    x = torch.tensor([-float('inf'), -1., 0., 1., float('inf')])
    y = x / x.unsqueeze(-1)
    mask = ~(y == y)
    f = torch.jit.trace(f, (y, mask))


def test_masked_fill_workaround():

    def f(y, mask):
        return y.clone().masked_fill_(mask, 0.)

    def g(y, mask):
        y = y.clone()
        y[mask] = 0.  # this is much slower than .masked_fill_()
        return y

    x = torch.tensor([-float('inf'), -1., 0., 1., float('inf')])
    y = x / x.unsqueeze(-1)
    mask = ~(y == y)
    assert_equal(f(y, mask), g(y, mask))
    g = torch.jit.trace(g, (y, mask))
    assert_equal(f(y, mask), g(y, mask))


@pytest.mark.xfail(reason="https://github.com/pytorch/pytorch/issues/11614")
def test_scatter():

    def make_one_hot(x, i):
        return x.new_zeros(x.shape).scatter(-1, i.unsqueeze(-1), 1.0)

    x = torch.randn(5, 4, 3)
    i = torch.randint(0, 3, torch.Size((5, 4)))
    torch.jit.trace(make_one_hot, (x, i))


@pytest.mark.filterwarnings('ignore:Converting a tensor to a Python integer')
def test_scatter_workaround():

    def make_one_hot_expected(x, i):
        return x.new_zeros(x.shape).scatter(-1, i.unsqueeze(-1), 1.0)

    def make_one_hot_actual(x, i):
        eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        return eye[i].clone()

    x = torch.randn(5, 4, 3)
    i = torch.randint(0, 3, torch.Size((5, 4)))
    torch.jit.trace(make_one_hot_actual, (x, i))
    expected = make_one_hot_expected(x, i)
    actual = make_one_hot_actual(x, i)
    assert_equal(actual, expected)


@pytest.mark.parametrize('expand', [False, True])
@pytest.mark.parametrize('shape', [(), (4,), (5, 4)])
@pytest.mark.filterwarnings('ignore:Converting a tensor to a Python boolean')
def test_bernoulli_enumerate(shape, expand):
    shape = torch.Size(shape)
    probs = torch.empty(shape).fill_(0.25)

    @pyro.ops.jit.trace
    def f(probs):
        d = dist.Bernoulli(probs)
        support = d.enumerate_support(expand=expand)
        return d.log_prob(support)

    log_prob = f(probs)
    assert log_prob.shape == (2,) + shape


@pytest.mark.parametrize('expand', [False, True])
@pytest.mark.parametrize('shape', [(3,), (4, 3), (5, 4, 3)])
def test_categorical_enumerate(shape, expand):
    shape = torch.Size(shape)
    probs = torch.ones(shape)

    @pyro.ops.jit.trace
    def f(probs):
        d = dist.Categorical(probs)
        support = d.enumerate_support(expand=expand)
        return d.log_prob(support)

    log_prob = f(probs)
    batch_shape = shape[:-1]
    assert log_prob.shape == shape[-1:] + batch_shape


@pytest.mark.parametrize('expand', [False, True])
@pytest.mark.parametrize('shape', [(3,), (4, 3), (5, 4, 3)])
@pytest.mark.filterwarnings('ignore:Converting a tensor to a Python integer')
def test_one_hot_categorical_enumerate(shape, expand):
    shape = torch.Size(shape)
    probs = torch.ones(shape)

    @pyro.ops.jit.trace
    def f(probs):
        d = dist.OneHotCategorical(probs)
        support = d.enumerate_support(expand=expand)
        return d.log_prob(support)

    log_prob = f(probs)
    batch_shape = shape[:-1]
    assert log_prob.shape == shape[-1:] + batch_shape


@pytest.mark.parametrize('num_particles', [1, 10])
@pytest.mark.parametrize('Elbo', [
    Trace_ELBO,
    JitTrace_ELBO,
    TraceGraph_ELBO,
    JitTraceGraph_ELBO,
    TraceEnum_ELBO,
    xfail_param(JitTraceEnum_ELBO, reason='einsum not supported in jit'),
])
def test_svi(Elbo, num_particles):
    pyro.clear_param_store()
    data = torch.arange(10.)

    def model(data):
        loc = pyro.param("loc", constant(0.0))
        scale = pyro.param("scale", constant(1.0), constraint=constraints.positive)
        pyro.sample("x", dist.Normal(loc, scale).expand_by(data.shape).independent(1), obs=data)

    def guide(data):
        pass

    elbo = Elbo(num_particles=num_particles, strict_enumeration_warning=False)
    inference = SVI(model, guide, Adam({"lr": 1e-6}), elbo)
    for i in range(100):
        inference.step(data)


@pytest.mark.parametrize("enumerate2", ["sequential", "parallel"])
@pytest.mark.parametrize("enumerate1", ["sequential", "parallel"])
@pytest.mark.parametrize("irange_dim", [1, 2])
@pytest.mark.parametrize('Elbo', [JitTraceEnum_ELBO])
def test_svi_enum(Elbo, irange_dim, enumerate1, enumerate2):
    pyro.clear_param_store()
    num_particles = 10
    q = pyro.param("q", constant(0.75), constraint=constraints.unit_interval)
    p = 0.2693204236205713  # for which kl(Bernoulli(q), Bernoulli(p)) = 0.5

    def model():
        pyro.sample("x", dist.Bernoulli(p))
        for i in pyro.irange("irange", irange_dim):
            pyro.sample("y_{}".format(i), dist.Bernoulli(p))

    def guide():
        q = pyro.param("q")
        pyro.sample("x", dist.Bernoulli(q), infer={"enumerate": enumerate1})
        for i in pyro.irange("irange", irange_dim):
            pyro.sample("y_{}".format(i), dist.Bernoulli(q), infer={"enumerate": enumerate2})

    kl = (1 + irange_dim) * kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p))
    expected_loss = kl.item()
    expected_grad = grad(kl, [q.unconstrained()])[0]

    inner_particles = 2
    outer_particles = num_particles // inner_particles
    elbo = Elbo(max_plate_nesting=0,
                strict_enumeration_warning=any([enumerate1, enumerate2]),
                num_particles=inner_particles,
                ignore_jit_warnings=True)
    actual_loss = sum(elbo.loss_and_grads(model, guide)
                      for i in range(outer_particles)) / outer_particles
    actual_grad = q.unconstrained().grad / outer_particles

    assert_equal(actual_loss, expected_loss, prec=0.3, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=0.5, msg="".join([
        "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize('vectorized', [False, True])
@pytest.mark.parametrize('Elbo', [TraceEnum_ELBO, JitTraceEnum_ELBO])
def test_beta_bernoulli(Elbo, vectorized):
    pyro.clear_param_store()
    data = torch.tensor([1.0] * 6 + [0.0] * 4)

    def model1(data):
        alpha0 = constant(10.0)
        beta0 = constant(10.0)
        f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
        for i in pyro.irange("irange", len(data)):
            pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

    def model2(data):
        alpha0 = constant(10.0)
        beta0 = constant(10.0)
        f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
        pyro.sample("obs", dist.Bernoulli(f).expand_by(data.shape).independent(1),
                    obs=data)

    model = model2 if vectorized else model1

    def guide(data):
        alpha_q = pyro.param("alpha_q", constant(15.0),
                             constraint=constraints.positive)
        beta_q = pyro.param("beta_q", constant(15.0),
                            constraint=constraints.positive)
        pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

    elbo = Elbo(num_particles=7, strict_enumeration_warning=False, ignore_jit_warnings=True)
    optim = Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
    svi = SVI(model, guide, optim, elbo)
    for step in range(40):
        svi.step(data)


@pytest.mark.parametrize('Elbo', [
    Trace_ELBO,
    JitTrace_ELBO,
    TraceGraph_ELBO,
    JitTraceGraph_ELBO,
    TraceEnum_ELBO,
    JitTraceEnum_ELBO,
])
def test_svi_irregular_batch_size(Elbo):
    pyro.clear_param_store()

    @poutine.broadcast
    def model(data):
        loc = pyro.param("loc", constant(0.0))
        scale = pyro.param("scale", constant(1.0), constraint=constraints.positive)
        with pyro.plate("data", data.shape[0]):
            pyro.sample("x",
                        dist.Normal(loc, scale).expand([data.shape[0]]),
                        obs=data)

    def guide(data):
        pass

    pyro.clear_param_store()
    elbo = Elbo(strict_enumeration_warning=False, max_plate_nesting=1)
    inference = SVI(model, guide, Adam({"lr": 1e-6}), elbo)
    inference.step(torch.ones(10))
    inference.step(torch.ones(3))


@pytest.mark.parametrize('vectorized', [False, True])
@pytest.mark.parametrize('Elbo', [TraceEnum_ELBO, JitTraceEnum_ELBO])
def test_dirichlet_bernoulli(Elbo, vectorized):
    pyro.clear_param_store()
    data = torch.tensor([1.0] * 6 + [0.0] * 4)

    def model1(data):
        concentration0 = constant([10.0, 10.0])
        f = pyro.sample("latent_fairness", dist.Dirichlet(concentration0))[1]
        for i in pyro.irange("irange", len(data)):
            pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

    def model2(data):
        concentration0 = constant([10.0, 10.0])
        f = pyro.sample("latent_fairness", dist.Dirichlet(concentration0))[1]
        pyro.sample("obs", dist.Bernoulli(f).expand_by(data.shape).independent(1),
                    obs=data)

    model = model2 if vectorized else model1

    def guide(data):
        concentration_q = pyro.param("concentration_q", constant([15.0, 15.0]),
                                     constraint=constraints.positive)
        pyro.sample("latent_fairness", dist.Dirichlet(concentration_q))

    elbo = Elbo(num_particles=7, strict_enumeration_warning=False, ignore_jit_warnings=True)
    optim = Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
    svi = SVI(model, guide, optim, elbo)
    for step in range(40):
        svi.step(data)


@pytest.mark.parametrize("x,y", [
    (CondIndepStackFrame("a", -1, torch.tensor(2000), 2), CondIndepStackFrame("a", -1, 2000, 2)),
    (CondIndepStackFrame("a", -1, 1, 2), CondIndepStackFrame("a", -1, torch.tensor(1), 2)),
])
def test_cond_indep_equality(x, y):
    assert x == y
    assert not x != y
    assert hash(x) == hash(y)


def test_jit_arange_workaround():
    def fn(x):
        y = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
        return torch.cumsum(y, 0) - 1

    compiled = torch.jit.trace(fn, torch.ones(3))
    assert_equal(compiled(torch.ones(10)), torch.arange(10))
