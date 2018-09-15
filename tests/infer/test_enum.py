from __future__ import absolute_import, division, print_function

import logging
import math
import os
import timeit
from collections import defaultdict

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints, kl_divergence

import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.poutine as poutine
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedGamma
from pyro.infer import SVI, config_enumerate
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.traceenum_elbo import TraceEnum_ELBO
from pyro.infer.util import LAST_CACHE_SIZE
from pyro.util import torch_isnan
from tests.common import assert_equal, skipif_param

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2

logger = logging.getLogger(__name__)


def _skip_cuda(*args):
    return skipif_param(*args,
                        condition="CUDA_TEST" in os.environ,
                        reason="https://github.com/uber/pyro/issues/1380")


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_order(depth, graph_type):

    @config_enumerate
    def model(depth):
        for i in range(depth):
            pyro.sample("x{}".format(i), dist.Bernoulli(0.5))

    traces = list(iter_discrete_traces(graph_type, model, depth))

    assert len(traces) == 2 ** depth
    for trace in traces:
        sites = [name for name, site in trace.nodes.items() if site["type"] == "sample"]
        assert sites == ["x{}".format(i) for i in range(depth)]


@pytest.mark.parametrize("graph_type", ["flat", "dense"])
def test_iter_discrete_traces_scalar(graph_type):
    pyro.clear_param_store()

    @config_enumerate
    def model():
        p = pyro.param("p", torch.tensor(0.05))
        probs = pyro.param("probs", torch.tensor([0.1, 0.2, 0.3, 0.4]))
        x = pyro.sample("x", dist.Bernoulli(p))
        y = pyro.sample("y", dist.Categorical(probs))
        return dict(x=x, y=y)

    traces = list(iter_discrete_traces(graph_type, model))

    probs = pyro.param("probs")
    assert len(traces) == 2 * len(probs)


@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("expand", [False, True])
def test_iter_discrete_traces_vector(expand, graph_type):
    pyro.clear_param_store()

    @config_enumerate(expand=expand)
    def model():
        p = pyro.param("p", torch.tensor([0.05, 0.15]))
        probs = pyro.param("probs", torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                                  [0.4, 0.3, 0.2, 0.1]]))
        with pyro.iarange("iarange", 2):
            x = pyro.sample("x", dist.Bernoulli(p))
            y = pyro.sample("y", dist.Categorical(probs))
            if expand:
                assert x.size() == (2,)
                assert y.size() == (2,)
            else:
                assert x.shape == (1,)
                assert y.shape == (1,)
        return dict(x=x, y=y)

    traces = list(iter_discrete_traces(graph_type, model))

    probs = pyro.param("probs")
    assert len(traces) == 2 * probs.size(-1)


# The usual dist.Bernoulli avoids NANs by clamping log prob. This unsafe version
# allows us to test additional NAN avoidance in _compute_dice_elbo().
class UnsafeBernoulli(dist.Bernoulli):
    def log_prob(self, value):
        i = value.long()
        j = torch.arange(len(self.probs), dtype=torch.long)
        return torch.stack([(-self.probs).log1p(), self.probs.log()])[i, j]


@pytest.mark.parametrize('sample_shape', [(), (2,), (3, 4)])
def test_unsafe_bernoulli(sample_shape):
    logits = torch.randn(10)
    p = dist.Bernoulli(logits=logits)
    q = UnsafeBernoulli(logits=logits)
    x = p.sample(sample_shape)
    assert_equal(p.log_prob(x), q.log_prob(x))


@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_avoid_nan(enumerate1):
    pyro.clear_param_store()

    def model():
        p = torch.tensor([0.0, 0.5, 1.0])
        with pyro.iarange("batch", 3):
            pyro.sample("z", UnsafeBernoulli(p))

    @config_enumerate(default=enumerate1)
    def guide():
        p = pyro.param("p", torch.tensor([0.0, 0.5, 1.0], requires_grad=True))
        with pyro.iarange("batch", 3):
            pyro.sample("z", UnsafeBernoulli(p))

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))
    loss = elbo.loss(model, guide)
    assert not math.isnan(loss), loss
    loss = elbo.differentiable_loss(model, guide)
    assert not torch_isnan(loss), loss
    loss = elbo.loss_and_grads(model, guide)
    assert not math.isnan(loss), loss


# A simple Gaussian mixture model, with no vectorization.
def gmm_model(data, verbose=False):
    p = pyro.param("p", torch.tensor(0.3, requires_grad=True))
    scale = pyro.param("scale", torch.tensor(1.0, requires_grad=True))
    mus = torch.tensor([-1.0, 1.0])
    for i in pyro.irange("data", len(data)):
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        z = z.long()
        if verbose:
            logger.debug("M{} z_{} = {}".format("  " * i, i, z.cpu().numpy()))
        pyro.sample("x_{}".format(i), dist.Normal(mus[z], scale), obs=data[i])


def gmm_guide(data, verbose=False):
    for i in pyro.irange("data", len(data)):
        p = pyro.param("p_{}".format(i), torch.tensor(0.6, requires_grad=True))
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        z = z.long()
        if verbose:
            logger.debug("G{} z_{} = {}".format("  " * i, i, z.cpu().numpy()))


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_model, gmm_guide])
def test_gmm_iter_discrete_traces(data_size, graph_type, model):
    pyro.clear_param_store()
    data = torch.arange(0., float(data_size))
    model = config_enumerate(model)
    traces = list(iter_discrete_traces(graph_type, model, data=data, verbose=True))
    # This non-vectorized version is exponential in data_size:
    assert len(traces) == 2**data_size


# A Gaussian mixture model, with vectorized batching.
def gmm_batch_model(data):
    p = pyro.param("p", torch.tensor([0.3], requires_grad=True))
    p = torch.cat([p, 1 - p])
    scale = pyro.param("scale", torch.tensor([1.0], requires_grad=True))
    mus = torch.tensor([-1.0, 1.0])
    with pyro.iarange("data", len(data)) as batch:
        n = len(batch)
        z = pyro.sample("z", dist.OneHotCategorical(p).expand_by([n]))
        assert z.shape[-1] == 2
        loc = (z * mus).sum(-1)
        pyro.sample("x", dist.Normal(loc, scale.expand(n)), obs=data[batch])


def gmm_batch_guide(data):
    with pyro.iarange("data", len(data)) as batch:
        n = len(batch)
        probs = pyro.param("probs", torch.tensor(torch.ones(n, 1) * 0.6, requires_grad=True))
        probs = torch.cat([probs, 1 - probs], dim=1)
        z = pyro.sample("z", dist.OneHotCategorical(probs))
        assert z.shape[-1] == 2


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_batch_model, gmm_batch_guide])
def test_gmm_batch_iter_discrete_traces(model, data_size, graph_type):
    pyro.clear_param_store()
    data = torch.arange(0., float(data_size))
    model = config_enumerate(model)
    traces = list(iter_discrete_traces(graph_type, model, data=data))
    # This vectorized version is independent of data_size:
    assert len(traces) == 2


@pytest.mark.parametrize("model,guide", [
    (gmm_model, gmm_guide),
    (gmm_batch_model, gmm_batch_guide),
], ids=["single", "batch"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_svi_step_smoke(model, guide, enumerate1):
    pyro.clear_param_store()
    data = torch.tensor([0.0, 1.0, 9.0])

    guide = config_enumerate(guide, default=enumerate1)
    optimizer = pyro.optim.Adam({"lr": .001})
    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))
    inference = SVI(model, guide, optimizer, loss=elbo)
    inference.step(data)


@pytest.mark.parametrize("model,guide", [
    (gmm_model, gmm_guide),
    (gmm_batch_model, gmm_batch_guide),
], ids=["single", "batch"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_differentiable_loss(model, guide, enumerate1):
    pyro.clear_param_store()
    data = torch.tensor([0.0, 1.0, 9.0])

    guide = config_enumerate(guide, default=enumerate1)
    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))

    pyro.set_rng_seed(0)
    loss = elbo.differentiable_loss(model, guide, data)
    param_names = sorted(pyro.get_param_store().get_all_param_names())
    actual_loss = loss.item()
    actual_grads = grad(loss, [pyro.param(name).unconstrained() for name in param_names])

    pyro.set_rng_seed(0)
    expected_loss = elbo.loss_and_grads(model, guide, data)
    expected_grads = [pyro.param(name).unconstrained().grad for name in param_names]

    assert_equal(actual_loss, expected_loss)
    for name, actual_grad, expected_grad in zip(param_names, actual_grads, expected_grads):
        assert_equal(actual_grad, expected_grad, msg='bad {} gradient. Expected:\n{}\nActual:\n{}'.format(
            name, expected_grad, actual_grad))


@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_svi_step_guide_uses_grad(enumerate1):
    data = torch.tensor([0., 1., 3.])

    @poutine.broadcast
    def model():
        scale = pyro.param("scale")
        loc = pyro.sample("loc", dist.Normal(0., 10.))
        pyro.sample("b", dist.Bernoulli(0.5))
        with pyro.iarange("data", len(data)):
            pyro.sample("obs", dist.Normal(loc, scale), obs=data)

    @config_enumerate(default=enumerate1)
    def guide():
        p = pyro.param("p", torch.tensor(0.5), constraint=constraints.unit_interval)
        scale = pyro.param("scale", torch.tensor(1.0), constraint=constraints.positive)
        var = pyro.param("var", torch.tensor(1.0), constraint=constraints.positive)

        x = torch.tensor(0., requires_grad=True)
        prior = dist.Normal(0., 10.).log_prob(x)
        likelihood = dist.Normal(x, scale).log_prob(data).sum()
        loss = -(prior + likelihood)
        g = grad(loss, [x], create_graph=True)[0]
        H = grad(g, [x], create_graph=True)[0]
        loc = x.detach() - g / H  # newton step
        pyro.sample("loc", dist.Normal(loc, var))
        pyro.sample("b", dist.Bernoulli(p))

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))
    inference = SVI(model, guide, pyro.optim.Adam({}), elbo)
    inference.step()


@pytest.mark.parametrize('scale', [1, 10])
@pytest.mark.parametrize("method", ["loss", "differentiable_loss", "loss_and_grads"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_elbo_bern(method, enumerate1, scale):
    pyro.clear_param_store()
    num_particles = 1 if enumerate1 else 10000
    prec = 0.001 if enumerate1 else 0.2
    q = pyro.param("q", torch.tensor(0.5, requires_grad=True))
    kl = kl_divergence(dist.Bernoulli(q), dist.Bernoulli(0.25))

    @poutine.scale(scale=scale)
    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Bernoulli(0.25).expand_by([num_particles]))

    @config_enumerate(default=enumerate1)
    @poutine.scale(scale=scale)
    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Bernoulli(q).expand_by([num_particles]))

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))

    if method == "loss":
        actual = elbo.loss(model, guide) / num_particles
        expected = kl.item() * scale
        assert_equal(actual, expected, prec=prec, msg="".join([
            "\nexpected = {}".format(expected),
            "\n  actual = {}".format(actual),
        ]))
    else:
        if method == "differentiable_loss":
            loss = elbo.differentiable_loss(model, guide)
            actual = grad(loss, [q])[0] / num_particles
        elif method == "loss_and_grads":
            elbo.loss_and_grads(model, guide)
            actual = q.grad / num_particles
        expected = grad(kl, [q])[0] * scale
        assert_equal(actual, expected, prec=prec, msg="".join([
            "\nexpected = {}".format(expected.detach().cpu().numpy()),
            "\n  actual = {}".format(actual.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("method", ["loss", "differentiable_loss", "loss_and_grads"])
@pytest.mark.parametrize("enumerate1", [None, "parallel"])
def test_elbo_normal(method, enumerate1):
    pyro.clear_param_store()
    num_particles = 1 if enumerate1 else 10000
    prec = 0.01
    q = pyro.param("q", torch.tensor(1., requires_grad=True))
    kl = kl_divergence(dist.Normal(q, 1.), dist.Normal(0., 1.))

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Normal(0., 1.).expand_by([num_particles]))

    @config_enumerate(default=enumerate1, num_samples=10000)
    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Normal(q, 1.).expand_by([num_particles]))

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))

    if method == "loss":
        actual = elbo.loss(model, guide) / num_particles
        expected = kl.item()
        assert_equal(actual, expected, prec=prec, msg="".join([
            "\nexpected = {}".format(expected),
            "\n  actual = {}".format(actual),
        ]))
    else:
        if method == "differentiable_loss":
            loss = elbo.differentiable_loss(model, guide)
            actual = grad(loss, [q])[0] / num_particles
        elif method == "loss_and_grads":
            elbo.loss_and_grads(model, guide)
            actual = q.grad / num_particles
        expected = grad(kl, [q])[0]
        assert_equal(actual, expected, prec=prec, msg="".join([
            "\nexpected = {}".format(expected.detach().cpu().numpy()),
            "\n  actual = {}".format(actual.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("enumerate1,num_samples1", [
    (None, None),
    ("sequential", None),
    ("parallel", None),
    ("parallel", 300),
])
@pytest.mark.parametrize("enumerate2,num_samples2", [
    (None, None),
    ("sequential", None),
    ("parallel", None),
    ("parallel", 300),
])
@pytest.mark.parametrize("method", ["differentiable_loss", "loss_and_grads"])
def test_elbo_bern_bern(method, enumerate1, enumerate2, num_samples1, num_samples2):
    pyro.clear_param_store()
    if enumerate1 and enumerate2 and num_samples1 is None and num_samples2 is None:
        num_particles = 1
        prec = 0.001
    else:
        num_particles = 2 * 300 * 300
        for n in [num_samples1, num_samples2]:
            if n is not None:
                num_particles = num_particles // n
        prec = 0.2

    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))

    def model():
        pyro.sample("x1", dist.Bernoulli(0.2))
        pyro.sample("x2", dist.Bernoulli(0.4))

    def guide():
        q = pyro.param("q")
        pyro.sample("x1", dist.Bernoulli(q), infer={"enumerate": enumerate1, "num_samples": num_samples1})
        pyro.sample("x2", dist.Bernoulli(q), infer={"enumerate": enumerate2, "num_samples": num_samples2})

    kl = sum(kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p)) for p in [0.2, 0.4])
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=0,
                          num_particles=num_particles,
                          vectorize_particles=True,
                          strict_enumeration_warning=any([enumerate1, enumerate2]))
    if method == "differentiable_loss":
        loss = elbo.differentiable_loss(model, guide)
        actual_loss = loss.item()
        actual_grad = grad(loss, [q])[0]
    else:
        actual_loss = elbo.loss_and_grads(model, guide)
        actual_grad = q.grad

    assert_equal(actual_loss, expected_loss, prec=prec, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=prec, msg="".join([
        "\nexpected grads = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grads = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate3", [None, "sequential", "parallel"])
@pytest.mark.parametrize("method", ["differentiable_loss", "loss_and_grads"])
def test_elbo_berns(method, enumerate1, enumerate2, enumerate3):
    pyro.clear_param_store()
    num_particles = 1 if all([enumerate1, enumerate2, enumerate3]) else 10000
    prec = 0.001 if all([enumerate1, enumerate2, enumerate3]) else 0.1
    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))

    def model():
        pyro.sample("x1", dist.Bernoulli(0.1))
        pyro.sample("x2", dist.Bernoulli(0.2))
        pyro.sample("x3", dist.Bernoulli(0.3))

    def guide():
        q = pyro.param("q")
        pyro.sample("x1", dist.Bernoulli(q), infer={"enumerate": enumerate1})
        pyro.sample("x2", dist.Bernoulli(q), infer={"enumerate": enumerate2})
        pyro.sample("x3", dist.Bernoulli(q), infer={"enumerate": enumerate3})

    kl = sum(kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p)) for p in [0.1, 0.2, 0.3])
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=0,
                          num_particles=num_particles,
                          vectorize_particles=True,
                          strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
    if method == "differentiable_loss":
        loss = elbo.differentiable_loss(model, guide)
        actual_loss = loss.item()
        actual_grad = grad(loss, [q])[0]
    else:
        actual_loss = elbo.loss_and_grads(model, guide)
        actual_grad = q.grad

    assert_equal(actual_loss, expected_loss, prec=prec, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=prec, msg="".join([
        "\nexpected grads = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grads = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("max_iarange_nesting", [0, 1])
@pytest.mark.parametrize("enumerate1", ["sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", ["sequential", "parallel"])
@pytest.mark.parametrize("enumerate3", ["sequential", "parallel"])
def test_elbo_categoricals(enumerate1, enumerate2, enumerate3, max_iarange_nesting):
    pyro.clear_param_store()
    p1 = torch.tensor([0.6, 0.4])
    p2 = torch.tensor([0.3, 0.3, 0.4])
    p3 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    q1 = pyro.param("q1", torch.tensor([0.4, 0.6], requires_grad=True))
    q2 = pyro.param("q2", torch.tensor([0.4, 0.3, 0.3], requires_grad=True))
    q3 = pyro.param("q3", torch.tensor([0.4, 0.3, 0.2, 0.1], requires_grad=True))

    def model():
        pyro.sample("x1", dist.Categorical(p1))
        pyro.sample("x2", dist.Categorical(p2))
        pyro.sample("x3", dist.Categorical(p3))

    def guide():
        pyro.sample("x1", dist.Categorical(pyro.param("q1")), infer={"enumerate": enumerate1})
        pyro.sample("x2", dist.Categorical(pyro.param("q2")), infer={"enumerate": enumerate2})
        pyro.sample("x3", dist.Categorical(pyro.param("q3")), infer={"enumerate": enumerate3})

    kl = (kl_divergence(dist.Categorical(q1), dist.Categorical(p1)) +
          kl_divergence(dist.Categorical(q2), dist.Categorical(p2)) +
          kl_divergence(dist.Categorical(q3), dist.Categorical(p3)))
    expected_loss = kl.item()
    expected_grads = grad(kl, [q1, q2, q3])

    elbo = TraceEnum_ELBO(max_iarange_nesting=max_iarange_nesting,
                          strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
    actual_loss = elbo.loss_and_grads(model, guide)
    actual_grads = [q1.grad, q2.grad, q3.grad]

    assert_equal(actual_loss, expected_loss, prec=0.001, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_equal(actual_grad, expected_grad, prec=0.001, msg="".join([
            "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
            "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("enumerate1", [None, "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "parallel"])
@pytest.mark.parametrize("enumerate3", [None, "parallel"])
@pytest.mark.parametrize("method", ["differentiable_loss", "loss_and_grads"])
def test_elbo_normals(method, enumerate1, enumerate2, enumerate3):
    pyro.clear_param_store()
    num_particles = 100 * 10 ** sum(1 for e in [enumerate1, enumerate2, enumerate3] if not e)
    prec = 0.1
    q = pyro.param("q", torch.tensor(0.0, requires_grad=True))

    def model():
        pyro.sample("x1", dist.Normal(0.25, 1.))
        pyro.sample("x2", dist.Normal(0.5, 1.))
        pyro.sample("x3", dist.Normal(1., 1.))

    def guide():
        q = pyro.param("q")
        pyro.sample("x1", dist.Normal(q, 1.), infer={"enumerate": enumerate1, "num_samples": 10})
        pyro.sample("x2", dist.Normal(q, 1.), infer={"enumerate": enumerate2, "num_samples": 10})
        pyro.sample("x3", dist.Normal(q, 1.), infer={"enumerate": enumerate3, "num_samples": 10})

    kl = sum(kl_divergence(dist.Normal(q, 1.), dist.Normal(p, 1.)) for p in [0.25, 0.5, 1.])
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=0,
                          num_particles=num_particles,
                          vectorize_particles=True,
                          strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
    if method == "differentiable_loss":
        loss = elbo.differentiable_loss(model, guide)
        actual_loss = loss.item()
        actual_grad = grad(loss, [q])[0]
    else:
        actual_loss = elbo.loss_and_grads(model, guide)
        actual_grad = q.grad

    assert_equal(actual_loss, expected_loss, prec=prec, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=prec, msg="".join([
        "\nexpected grads = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grads = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("iarange_dim", [1, 2])
def test_elbo_iarange(iarange_dim, enumerate1, enumerate2):
    pyro.clear_param_store()
    num_particles = 1 if all([enumerate1, enumerate2]) else 10000
    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))
    p = 0.2693204236205713  # for which kl(Bernoulli(q), Bernoulli(p)) = 0.5

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("y", dist.Bernoulli(p).expand_by([num_particles]))
            with pyro.iarange("iarange", iarange_dim):
                pyro.sample("z", dist.Bernoulli(p).expand_by([iarange_dim, num_particles]))

    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("y", dist.Bernoulli(q).expand_by([num_particles]),
                        infer={"enumerate": enumerate1})
            with pyro.iarange("iarange", iarange_dim):
                pyro.sample("z", dist.Bernoulli(q).expand_by([iarange_dim, num_particles]),
                            infer={"enumerate": enumerate2})

    kl = (1 + iarange_dim) * kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p))
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=2,
                          strict_enumeration_warning=any([enumerate1, enumerate2]))
    actual_loss = elbo.loss_and_grads(model, guide) / num_particles
    actual_grad = pyro.param('q').grad / num_particles

    assert_equal(actual_loss, expected_loss, prec=0.1, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("irange_dim", [1, 2])
def test_elbo_irange(irange_dim, enumerate1, enumerate2):
    pyro.clear_param_store()
    num_particles = 1 if all([enumerate1, enumerate2]) else 10000
    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))
    p = 0.2693204236205713  # for which kl(Bernoulli(q), Bernoulli(p)) = 0.5

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(p).expand_by([num_particles]))
            for i in pyro.irange("irange", irange_dim):
                pyro.sample("y_{}".format(i), dist.Bernoulli(p).expand_by([num_particles]))

    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(q).expand_by([num_particles]),
                        infer={"enumerate": enumerate1})
            for i in pyro.irange("irange", irange_dim):
                pyro.sample("y_{}".format(i), dist.Bernoulli(q).expand_by([num_particles]),
                            infer={"enumerate": enumerate2})

    kl = (1 + irange_dim) * kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p))
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1, enumerate2]))
    actual_loss = elbo.loss_and_grads(model, guide) / num_particles
    actual_grad = pyro.param('q').grad / num_particles

    assert_equal(actual_loss, expected_loss, prec=0.1, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate4", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate3", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
def test_elbo_iarange_iarange(outer_dim, inner_dim, enumerate1, enumerate2, enumerate3, enumerate4):
    pyro.clear_param_store()
    num_particles = 1 if all([enumerate1, enumerate2, enumerate3, enumerate4]) else 100000
    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))
    p = 0.2693204236205713  # for which kl(Bernoulli(q), Bernoulli(p)) = 0.5

    @poutine.broadcast
    def model():
        d = dist.Bernoulli(p)
        context1 = pyro.iarange("outer", outer_dim, dim=-1)
        context2 = pyro.iarange("inner", inner_dim, dim=-2)
        pyro.sample("w", d)
        with context1:
            pyro.sample("x", d)
        with context2:
            pyro.sample("y", d)
        with context1, context2:
            pyro.sample("z", d)

    @poutine.broadcast
    def guide():
        d = dist.Bernoulli(pyro.param("q"))
        context1 = pyro.iarange("outer", outer_dim, dim=-1)
        context2 = pyro.iarange("inner", inner_dim, dim=-2)
        pyro.sample("w", d, infer={"enumerate": enumerate1})
        with context1:
            pyro.sample("x", d, infer={"enumerate": enumerate2})
        with context2:
            pyro.sample("y", d, infer={"enumerate": enumerate3})
        with context1, context2:
            pyro.sample("z", d, infer={"enumerate": enumerate4})

    kl_node = kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p))
    kl = (1 + outer_dim + inner_dim + outer_dim * inner_dim) * kl_node
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=3,
                          num_particles=num_particles,
                          vectorize_particles=True,
                          strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
    actual_loss = elbo.loss_and_grads(model, guide)
    actual_grad = pyro.param('q').grad

    assert_equal(actual_loss, expected_loss, prec=0.1, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate3", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [3])
def test_elbo_iarange_irange(outer_dim, inner_dim, enumerate1, enumerate2, enumerate3):
    pyro.clear_param_store()
    num_particles = 1 if all([enumerate1, enumerate2, enumerate3]) else 100000
    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))
    p = 0.2693204236205713  # for which kl(Bernoulli(q), Bernoulli(p)) = 0.5

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(p).expand_by([num_particles]))
            with pyro.iarange("outer", outer_dim):
                pyro.sample("y", dist.Bernoulli(p).expand_by([outer_dim, num_particles]))
                for i in pyro.irange("inner", inner_dim):
                    pyro.sample("z_{}".format(i), dist.Bernoulli(p).expand_by([outer_dim, num_particles]))

    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(q).expand_by([num_particles]),
                        infer={"enumerate": enumerate1})
            with pyro.iarange("outer", outer_dim):
                pyro.sample("y", dist.Bernoulli(q).expand_by([outer_dim, num_particles]),
                            infer={"enumerate": enumerate2})
                for i in pyro.irange("inner", inner_dim):
                    pyro.sample("z_{}".format(i), dist.Bernoulli(q).expand_by([outer_dim, num_particles]),
                                infer={"enumerate": enumerate3})

    kl = (1 + outer_dim * (1 + inner_dim)) * kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p))
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=2,
                          strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
    actual_loss = elbo.loss_and_grads(model, guide) / num_particles
    actual_grad = pyro.param('q').grad / num_particles

    assert_equal(actual_loss, expected_loss, prec=0.1, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate3", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
def test_elbo_irange_iarange(outer_dim, inner_dim, enumerate1, enumerate2, enumerate3):
    pyro.clear_param_store()
    num_particles = 1 if all([enumerate1, enumerate2, enumerate3]) else 50000
    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))
    p = 0.2693204236205713  # for which kl(Bernoulli(q), Bernoulli(p)) = 0.5

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(p).expand_by([num_particles]))
            inner_iarange = pyro.iarange("inner", inner_dim)
            for i in pyro.irange("outer", outer_dim):
                pyro.sample("y_{}".format(i), dist.Bernoulli(p).expand_by([num_particles]))
                with inner_iarange:
                    pyro.sample("z_{}".format(i), dist.Bernoulli(p).expand_by([inner_dim, num_particles]))

    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(q).expand_by([num_particles]),
                        infer={"enumerate": enumerate1})
            inner_iarange = pyro.iarange("inner", inner_dim)
            for i in pyro.irange("outer", outer_dim):
                pyro.sample("y_{}".format(i), dist.Bernoulli(q).expand_by([num_particles]),
                            infer={"enumerate": enumerate2})
                with inner_iarange:
                    pyro.sample("z_{}".format(i), dist.Bernoulli(q).expand_by([inner_dim, num_particles]),
                                infer={"enumerate": enumerate3})

    kl = (1 + outer_dim * (1 + inner_dim)) * kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p))
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=2,
                          strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
    actual_loss = elbo.loss_and_grads(model, guide) / num_particles
    actual_grad = pyro.param('q').grad / num_particles

    assert_equal(actual_loss, expected_loss, prec=0.1, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate3", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
def test_elbo_irange_irange(outer_dim, inner_dim, enumerate1, enumerate2, enumerate3):
    pyro.clear_param_store()
    num_particles = 1 if all([enumerate1, enumerate2, enumerate3]) else 50000
    q = pyro.param("q", torch.tensor(0.75, requires_grad=True))
    p = 0.2693204236205713  # for which kl(Bernoulli(q), Bernoulli(p)) = 0.5

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(p).expand_by([num_particles]))
            inner_irange = pyro.irange("inner", outer_dim)
            for i in pyro.irange("outer", inner_dim):
                pyro.sample("y_{}".format(i), dist.Bernoulli(p).expand_by([num_particles]))
                for j in inner_irange:
                    pyro.sample("z_{}_{}".format(i, j), dist.Bernoulli(p).expand_by([num_particles]))

    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("x", dist.Bernoulli(q).expand_by([num_particles]),
                        infer={"enumerate": enumerate1})
            inner_irange = pyro.irange("inner", inner_dim)
            for i in pyro.irange("outer", outer_dim):
                pyro.sample("y_{}".format(i), dist.Bernoulli(q).expand_by([num_particles]),
                            infer={"enumerate": enumerate2})
                for j in inner_irange:
                    pyro.sample("z_{}_{}".format(i, j), dist.Bernoulli(q).expand_by([num_particles]),
                                infer={"enumerate": enumerate3})

    kl = (1 + outer_dim * (1 + inner_dim)) * kl_divergence(dist.Bernoulli(q), dist.Bernoulli(p))
    expected_loss = kl.item()
    expected_grad = grad(kl, [q])[0]

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1, enumerate2, enumerate3]))
    actual_loss = elbo.loss_and_grads(model, guide) / num_particles
    actual_grad = pyro.param('q').grad / num_particles

    assert_equal(actual_loss, expected_loss, prec=0.1, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))
    assert_equal(actual_grad, expected_grad, prec=0.1, msg="".join([
        "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
        "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("pi1", [0.33, 0.43])
@pytest.mark.parametrize("pi2", [0.55, 0.27])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_non_mean_field_bern_bern_elbo_gradient(enumerate1, pi1, pi2):
    pyro.clear_param_store()
    num_particles = 1 if enumerate1 else 20000

    def model():
        with pyro.iarange("particles", num_particles):
            y = pyro.sample("y", dist.Bernoulli(0.33).expand_by([num_particles]))
            pyro.sample("z", dist.Bernoulli(0.55 * y + 0.10))

    def guide():
        q1 = pyro.param("q1", torch.tensor(pi1, requires_grad=True))
        q2 = pyro.param("q2", torch.tensor(pi2, requires_grad=True))
        with pyro.iarange("particles", num_particles):
            y = pyro.sample("y", dist.Bernoulli(q1).expand_by([num_particles]))
            pyro.sample("z", dist.Bernoulli(q2 * y + 0.10))

    logger.info("Computing gradients using surrogate loss")
    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))
    elbo.loss_and_grads(model, config_enumerate(guide, default=enumerate1))
    actual_grad_q1 = pyro.param('q1').grad / num_particles
    actual_grad_q2 = pyro.param('q2').grad / num_particles

    logger.info("Computing analytic gradients")
    q1 = torch.tensor(pi1, requires_grad=True)
    q2 = torch.tensor(pi2, requires_grad=True)
    elbo = kl_divergence(dist.Bernoulli(q1), dist.Bernoulli(0.33))
    elbo = elbo + q1 * kl_divergence(dist.Bernoulli(q2 + 0.10), dist.Bernoulli(0.65))
    elbo = elbo + (1.0 - q1) * kl_divergence(dist.Bernoulli(0.10), dist.Bernoulli(0.10))
    expected_grad_q1, expected_grad_q2 = grad(elbo, [q1, q2])

    prec = 0.03 if enumerate1 is None else 0.001

    assert_equal(actual_grad_q1, expected_grad_q1, prec=prec, msg="".join([
        "\nq1 expected = {}".format(expected_grad_q1.data.cpu().numpy()),
        "\nq1  actual = {}".format(actual_grad_q1.data.cpu().numpy()),
    ]))
    assert_equal(actual_grad_q2, expected_grad_q2, prec=prec, msg="".join([
        "\nq2 expected = {}".format(expected_grad_q2.data.cpu().numpy()),
        "\nq2   actual = {}".format(actual_grad_q2.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("pi1", [0.33, 0.44])
@pytest.mark.parametrize("pi2", [0.55, 0.39])
@pytest.mark.parametrize("pi3", [0.22, 0.29])
@pytest.mark.parametrize("enumerate1,num_samples", [
    (None, None),
    ("sequential", None),
    ("parallel", None),
    ("parallel", 2),
])
def test_non_mean_field_bern_normal_elbo_gradient(enumerate1, pi1, pi2, pi3, num_samples):
    pyro.clear_param_store()
    include_z = True
    num_particles = 10000

    def model():
        with pyro.iarange("particles", num_particles):
            q3 = pyro.param("q3", torch.tensor(pi3, requires_grad=True))
            y = pyro.sample("y", dist.Bernoulli(q3).expand_by([num_particles]))
            if include_z:
                pyro.sample("z", dist.Normal(0.55 * y + q3, 1.0))

    def guide():
        q1 = pyro.param("q1", torch.tensor(pi1, requires_grad=True))
        q2 = pyro.param("q2", torch.tensor(pi2, requires_grad=True))
        with pyro.iarange("particles", num_particles):
            y = pyro.sample("y", dist.Bernoulli(q1).expand_by([num_particles]), infer={"enumerate": enumerate1})
            if include_z:
                pyro.sample("z", dist.Normal(q2 * y + 0.10, 1.0))

    logger.info("Computing gradients using surrogate loss")
    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))
    elbo.loss_and_grads(model, guide)
    actual_grad_q1 = pyro.param('q1').grad / num_particles
    if include_z:
        actual_grad_q2 = pyro.param('q2').grad / num_particles
    actual_grad_q3 = pyro.param('q3').grad / num_particles

    logger.info("Computing analytic gradients")
    q1 = torch.tensor(pi1, requires_grad=True)
    q2 = torch.tensor(pi2, requires_grad=True)
    q3 = torch.tensor(pi3, requires_grad=True)
    elbo = kl_divergence(dist.Bernoulli(q1), dist.Bernoulli(q3))
    if include_z:
        elbo = elbo + q1 * kl_divergence(dist.Normal(q2 + 0.10, 1.0), dist.Normal(q3 + 0.55, 1.0))
        elbo = elbo + (1.0 - q1) * kl_divergence(dist.Normal(0.10, 1.0), dist.Normal(q3, 1.0))
        expected_grad_q1, expected_grad_q2, expected_grad_q3 = grad(elbo, [q1, q2, q3])
    else:
        expected_grad_q1, expected_grad_q3 = grad(elbo, [q1, q3])

    prec = 0.04 if enumerate1 is None else 0.02

    assert_equal(actual_grad_q1, expected_grad_q1, prec=prec, msg="".join([
        "\nq1 expected = {}".format(expected_grad_q1.data.cpu().numpy()),
        "\nq1   actual = {}".format(actual_grad_q1.data.cpu().numpy()),
    ]))
    if include_z:
        assert_equal(actual_grad_q2, expected_grad_q2, prec=prec, msg="".join([
            "\nq2 expected = {}".format(expected_grad_q2.data.cpu().numpy()),
            "\nq2   actual = {}".format(actual_grad_q2.data.cpu().numpy()),
        ]))
    assert_equal(actual_grad_q3, expected_grad_q3, prec=prec, msg="".join([
        "\nq3 expected = {}".format(expected_grad_q3.data.cpu().numpy()),
        "\nq3   actual = {}".format(actual_grad_q3.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("pi1", [0.33, 0.41])
@pytest.mark.parametrize("pi2", [0.44, 0.17])
@pytest.mark.parametrize("pi3", [0.22, 0.29])
def test_non_mean_field_normal_bern_elbo_gradient(pi1, pi2, pi3):

    def model(num_particles):
        with pyro.iarange("particles", num_particles):
            q3 = pyro.param("q3", torch.tensor(pi3, requires_grad=True))
            q4 = pyro.param("q4", torch.tensor(0.5 * (pi1 + pi2), requires_grad=True))
            z = pyro.sample("z", dist.Normal(q3, 1.0).expand_by([num_particles]))
            zz = torch.exp(z) / (1.0 + torch.exp(z))
            pyro.sample("y", dist.Bernoulli(q4 * zz))

    def guide(num_particles):
        q1 = pyro.param("q1", torch.tensor(pi1, requires_grad=True))
        q2 = pyro.param("q2", torch.tensor(pi2, requires_grad=True))
        with pyro.iarange("particles", num_particles):
            z = pyro.sample("z", dist.Normal(q2, 1.0).expand_by([num_particles]))
            zz = torch.exp(z) / (1.0 + torch.exp(z))
            pyro.sample("y", dist.Bernoulli(q1 * zz))

    qs = ['q1', 'q2', 'q3', 'q4']
    results = {}

    for ed, num_particles in zip([None, 'parallel', 'sequential'], [30000, 20000, 20000]):
        pyro.clear_param_store()
        elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                              strict_enumeration_warning=any([ed]))
        elbo.loss_and_grads(model, config_enumerate(guide, default=ed), num_particles)
        results[str(ed)] = {}
        for q in qs:
            results[str(ed)]['actual_grad_%s' % q] = pyro.param(q).grad.detach().cpu().numpy() / num_particles

    prec = 0.03
    for ed in ['parallel', 'sequential']:
        logger.info('\n*** {} ***'.format(ed))
        for q in qs:
            logger.info("[{}] actual: {}".format(q, results[ed]['actual_grad_%s' % q]))
            assert_equal(results[ed]['actual_grad_%s' % q], results['None']['actual_grad_%s' % q], prec=prec,
                         msg="".join([
                             "\nexpected (MC estimate) = {}".format(results['None']['actual_grad_%s' % q]),
                             "\n  actual ({} estimate) = {}".format(ed, results[ed]['actual_grad_%s' % q]),
                         ]))


@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_elbo_rsvi(enumerate1):
    pyro.clear_param_store()
    num_particles = 40000
    prec = 0.01 if enumerate1 else 0.02
    q = pyro.param("q", torch.tensor(0.5, requires_grad=True))
    a = pyro.param("a", torch.tensor(1.5, requires_grad=True))
    kl1 = kl_divergence(dist.Bernoulli(q), dist.Bernoulli(0.25))
    kl2 = kl_divergence(dist.Gamma(a, 1.0), dist.Gamma(0.5, 1.0))

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Bernoulli(0.25).expand_by([num_particles]))
            pyro.sample("y", dist.Gamma(0.50, 1.0).expand_by([num_particles]))

    @config_enumerate(default=enumerate1)
    def guide():
        q = pyro.param("q")
        a = pyro.param("a")
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Bernoulli(q).expand_by([num_particles]))
            pyro.sample("y", ShapeAugmentedGamma(a, torch.tensor(1.0)).expand_by([num_particles]))

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))
    elbo.loss_and_grads(model, guide)

    actual_q = q.grad / num_particles
    expected_q = grad(kl1, [q])[0]
    assert_equal(actual_q, expected_q, prec=prec, msg="".join([
        "\nexpected q.grad = {}".format(expected_q.detach().cpu().numpy()),
        "\n  actual q.grad = {}".format(actual_q.detach().cpu().numpy()),
    ]))
    actual_a = a.grad / num_particles
    expected_a = grad(kl2, [a])[0]
    assert_equal(actual_a, expected_a, prec=prec, msg="".join([
        "\nexpected a.grad= {}".format(expected_a.detach().cpu().numpy()),
        "\n  actual a.grad = {}".format(actual_a.detach().cpu().numpy()),
    ]))


@pytest.mark.parametrize("enumerate1,num_steps,expand", [
    ("sequential", 2, True),
    ("sequential", 2, False),
    ("sequential", 3, True),
    ("sequential", 3, False),
    ("parallel", 2, True),
    ("parallel", 2, False),
    ("parallel", 3, True),
    ("parallel", 3, False),
    ("parallel", 10, False),
    ("parallel", 20, False),
    _skip_cuda("parallel", 30, False),
])
def test_elbo_hmm_in_model(enumerate1, num_steps, expand):
    pyro.clear_param_store()
    data = torch.ones(num_steps)
    init_probs = torch.tensor([0.5, 0.5])

    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                                      constraint=constraints.simplex)
        locs = pyro.param("obs_locs", torch.tensor([-1.0, 1.0]))
        scale = pyro.param("obs_scale", torch.tensor(1.0),
                           constraint=constraints.positive)

        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))
            pyro.sample("y_{}".format(i), dist.Normal(locs[x], scale), obs=y)

    @config_enumerate(default=enumerate1, expand=expand)
    def guide(data):
        mean_field_probs = pyro.param("mean_field_probs", torch.ones(num_steps, 2) / 2,
                                      constraint=constraints.simplex)
        for i in range(num_steps):
            pyro.sample("x_{}".format(i), dist.Categorical(mean_field_probs[i]))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    elbo.loss_and_grads(model, guide, data)

    expected_unconstrained_grads = {
        "transition_probs": torch.tensor([[0.2, -0.2], [-0.2, 0.2]]) * (num_steps - 1),
        "obs_locs": torch.tensor([-num_steps, 0]),
        "obs_scale": torch.tensor(-num_steps),
        "mean_field_probs": torch.tensor([[0.5, -0.5]] * num_steps),
    }

    for name, value in pyro.get_param_store().named_parameters():
        actual = value.grad
        expected = expected_unconstrained_grads[name]
        assert_equal(actual, expected, msg=''.join([
            '\nexpected {}.grad = {}'.format(name, expected.cpu().numpy()),
            '\n  actual {}.grad = {}'.format(name, actual.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("enumerate1,num_steps,expand", [
    ("sequential", 2, True),
    ("sequential", 2, False),
    ("sequential", 3, True),
    ("sequential", 3, False),
    ("parallel", 2, True),
    ("parallel", 2, False),
    ("parallel", 3, True),
    ("parallel", 3, False),
    ("parallel", 10, False),
    ("parallel", 20, False),
    _skip_cuda("parallel", 30, False),
    _skip_cuda("parallel", 40, False),
    _skip_cuda("parallel", 50, False),
])
def test_elbo_hmm_in_guide(enumerate1, num_steps, expand):
    pyro.clear_param_store()
    data = torch.ones(num_steps)
    init_probs = torch.tensor([0.5, 0.5])

    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        emission_probs = pyro.param("emission_probs",
                                    torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                    constraint=constraints.simplex)
        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))
            pyro.sample("y_{}".format(i), dist.Categorical(emission_probs[x]), obs=y)

    @config_enumerate(default=enumerate1, expand=expand)
    def guide(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    elbo.loss_and_grads(model, guide, data)

    # These golden values simply test agreement between parallel and sequential.
    expected_grads = {
        2: {
            "transition_probs": [[0.1029949, -0.1029949], [0.1029949, -0.1029949]],
            "emission_probs": [[0.75, -0.75], [0.25, -0.25]],
        },
        3: {
            "transition_probs": [[0.25748726, -0.25748726], [0.25748726, -0.25748726]],
            "emission_probs": [[1.125, -1.125], [0.375, -0.375]],
        },
        10: {
            "transition_probs": [[1.64832076, -1.64832076], [1.64832076, -1.64832076]],
            "emission_probs": [[3.75, -3.75], [1.25, -1.25]],
        },
        20: {
            "transition_probs": [[3.70781687, -3.70781687], [3.70781687, -3.70781687]],
            "emission_probs": [[7.5, -7.5], [2.5, -2.5]],
        },
        22: {
            "transition_probs": [[4.11979618, -4.11979618], [4.11979618, -4.11979618]],
            "emission_probs": [[8.25, -8.25], [2.75, -2.75]],
        },
        30: {
            "transition_probs": [[5.76771452, -5.76771452], [5.76771452, -5.76771452]],
            "emission_probs": [[11.25, -11.25], [3.75, -3.75]],
        },
    }

    if num_steps not in expected_grads:
        return
    for name, value in pyro.get_param_store().named_parameters():
        actual = value.grad
        expected = torch.tensor(expected_grads[num_steps][name])
        assert_equal(actual, expected, msg=''.join([
            '\nexpected {}.grad = {}'.format(name, expected.cpu().numpy()),
            '\n  actual {}.grad = {}'.format(name, actual.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize('num_steps', [2, 3, 4, 5, 10, 20, _skip_cuda(30)])
def test_hmm_enumerate_model(num_steps):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))

    @config_enumerate(default="parallel")
    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        emission_probs = pyro.param("emission_probs",
                                    torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                    constraint=constraints.simplex)
        x = 0
        for t, y in enumerate(data):
            x = pyro.sample("x_{}".format(t), dist.Categorical(transition_probs[x]))
            pyro.sample("y_{}".format(t), dist.Categorical(emission_probs[x]), obs=y)
            print('{}\t{}'.format(t, tuple(x.shape)))

    def guide(data):
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    elbo.differentiable_loss(model, guide, data)


@pytest.mark.parametrize('num_steps', [2, 3, 4, 5, 10, 20, _skip_cuda(30)])
def test_hmm_enumerate_model_and_guide(num_steps):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))

    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        emission_probs = pyro.param("emission_probs",
                                    torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                    constraint=constraints.simplex)
        x = pyro.sample("x", dist.Categorical(torch.tensor([0.5, 0.5])))
        print('-1\t{}'.format(tuple(x.shape)))
        for t, y in enumerate(data):
            x = pyro.sample("x_{}".format(t), dist.Categorical(transition_probs[x]),
                            infer={"enumerate": "parallel"})
            pyro.sample("y_{}".format(t), dist.Categorical(emission_probs[x]), obs=y)
            print('{}\t{}'.format(t, tuple(x.shape)))

    def guide(data):
        init_probs = pyro.param("init_probs",
                                torch.tensor([0.75, 0.25]),
                                constraint=constraints.simplex)
        pyro.sample("x", dist.Categorical(init_probs),
                    infer={"enumerate": "parallel"})

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    elbo.differentiable_loss(model, guide, data)


def _check_loss_and_grads(expected_loss, actual_loss):
    assert_equal(actual_loss, expected_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(expected_loss.detach().cpu().numpy(),
                                                         actual_loss.detach().cpu().numpy()))

    names = pyro.get_param_store().get_all_param_names()
    params = [pyro.param(name).unconstrained() for name in names]
    actual_grads = grad(actual_loss, params, allow_unused=True, retain_graph=True)
    expected_grads = grad(expected_loss, params, allow_unused=True, retain_graph=True)
    for name, actual_grad, expected_grad in zip(names, actual_grads, expected_grads):
        if actual_grad is None or expected_grad is None:
            continue
        assert_equal(actual_grad, expected_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 expected_grad.detach().cpu().numpy(),
                                                                 actual_grad.detach().cpu().numpy()))


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_1(scale):
    pyro.param("guide_probs_x",
               torch.tensor([0.1, 0.9]),
               constraint=constraints.simplex)
    pyro.param("model_probs_x",
               torch.tensor([0.4, 0.6]),
               constraint=constraints.simplex)
    pyro.param("model_probs_y",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_z",
               torch.tensor([0.3, 0.7]),
               constraint=constraints.simplex)

    @poutine.scale(scale=scale)
    def auto_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("y", dist.Categorical(probs_y[x]),
                    infer={"enumerate": "parallel"})
        pyro.sample("z", dist.Categorical(probs_z), obs=torch.tensor(0))

    @poutine.scale(scale=scale)
    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_z = pyro.param("model_probs_z")
        pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("z", dist.Categorical(probs_z), obs=torch.tensor(0))

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_2(scale):
    pyro.param("guide_probs_x",
               torch.tensor([0.1, 0.9]),
               constraint=constraints.simplex)
    pyro.param("model_probs_x",
               torch.tensor([0.4, 0.6]),
               constraint=constraints.simplex)
    pyro.param("model_probs_y",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_z",
               torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
               constraint=constraints.simplex)

    @poutine.scale(scale=scale)
    def auto_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        y = pyro.sample("y", dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel"})
        pyro.sample("z", dist.Categorical(probs_z[y]), obs=torch.tensor(0))

    @poutine.scale(scale=scale)
    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        x = pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("z", dist.Categorical(probs_yz[x]), obs=torch.tensor(0))

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_3(scale):
    pyro.param("guide_probs_x",
               torch.tensor([0.1, 0.9]),
               constraint=constraints.simplex)
    pyro.param("model_probs_x",
               torch.tensor([0.4, 0.6]),
               constraint=constraints.simplex)
    pyro.param("model_probs_y",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_z",
               torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
               constraint=constraints.simplex)

    def auto_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            y = pyro.sample("y", dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel"})
            pyro.sample("z", dist.Categorical(probs_z[y]), obs=torch.tensor(0))

    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            pyro.sample("z", dist.Categorical(probs_yz[x]), obs=torch.tensor(0))

    @config_enumerate(default="parallel")
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
@pytest.mark.parametrize('num_samples,num_masked',
                         [(1, 1), (2, 2), (3, 2)],
                         ids=["single", "batch", "masked"])
def test_elbo_enumerate_iarange_1(num_samples, num_masked, scale):
    #              +---------+
    #  x ----> y ----> z     |
    #              |       N |
    #              +---------+
    pyro.param("guide_probs_x",
               torch.tensor([0.1, 0.9]),
               constraint=constraints.simplex)
    pyro.param("model_probs_x",
               torch.tensor([0.4, 0.6]),
               constraint=constraints.simplex)
    pyro.param("model_probs_y",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_z",
               torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
               constraint=constraints.simplex)

    def auto_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            y = pyro.sample("y", dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel"})
            if num_masked == num_samples:
                with pyro.iarange("data", len(data)):
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
            else:
                with pyro.iarange("data", len(data)):
                    with poutine.mask(mask=torch.arange(num_samples) < num_masked):
                        pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            y = pyro.sample("y", dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel"})
            for i in pyro.irange("data", num_masked):
                pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=data[i])

    @config_enumerate(default="parallel")
    def guide(data):
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
@pytest.mark.parametrize('num_samples,num_masked',
                         [(1, 1), (2, 2), (3, 2)],
                         ids=["single", "batch", "masked"])
def test_elbo_enumerate_iarange_2(num_samples, num_masked, scale):
    #      +-----------------+
    #  x ----> y ----> z     |
    #      |               N |
    #      +-----------------+
    pyro.param("guide_probs_x",
               torch.tensor([0.1, 0.9]),
               constraint=constraints.simplex)
    pyro.param("model_probs_x",
               torch.tensor([0.4, 0.6]),
               constraint=constraints.simplex)
    pyro.param("model_probs_y",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_z",
               torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
               constraint=constraints.simplex)

    def auto_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            with pyro.iarange("data", len(data)):
                if num_masked == num_samples:
                    y = pyro.sample("y", dist.Categorical(probs_y[x]),
                                    infer={"enumerate": "parallel"})
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
                else:
                    with poutine.mask(mask=torch.arange(num_samples) < num_masked):
                        y = pyro.sample("y", dist.Categorical(probs_y[x]),
                                        infer={"enumerate": "parallel"})
                        pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            for i in pyro.irange("data", num_masked):
                y = pyro.sample("y_{}".format(i), dist.Categorical(probs_y[x]),
                                infer={"enumerate": "parallel"})
                pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=data[i])

    @config_enumerate(default="parallel")
    def guide(data):
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
@pytest.mark.parametrize('num_samples,num_masked',
                         [(1, 1), (2, 2), (3, 2)],
                         ids=["single", "batch", "masked"])
def test_elbo_enumerate_iarange_3(num_samples, num_masked, scale):
    #  +-----------------------+
    #  | x ----> y ----> z     |
    #  |                     N |
    #  +-----------------------+
    # This iarange should remain unreduced since all enumeration is in a single iarange.
    pyro.param("guide_probs_x",
               torch.tensor([0.1, 0.9]),
               constraint=constraints.simplex)
    pyro.param("model_probs_x",
               torch.tensor([0.4, 0.6]),
               constraint=constraints.simplex)
    pyro.param("model_probs_y",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_z",
               torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
               constraint=constraints.simplex)

    @poutine.scale(scale=scale)
    def auto_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        with pyro.iarange("data", len(data)):
            if num_masked == num_samples:
                x = pyro.sample("x", dist.Categorical(probs_x))
                y = pyro.sample("y", dist.Categorical(probs_y[x]),
                                infer={"enumerate": "parallel"})
                pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
            else:
                with poutine.mask(mask=torch.arange(num_samples) < num_masked):
                    x = pyro.sample("x", dist.Categorical(probs_x))
                    y = pyro.sample("y", dist.Categorical(probs_y[x]),
                                    infer={"enumerate": "parallel"})
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    @poutine.scale(scale=scale)
    @config_enumerate(default="parallel")
    def auto_guide(data):
        probs_x = pyro.param("guide_probs_x")
        with pyro.iarange("data", len(data)):
            if num_masked == num_samples:
                pyro.sample("x", dist.Categorical(probs_x))
            else:
                with poutine.mask(mask=torch.arange(num_samples) < num_masked):
                    pyro.sample("x", dist.Categorical(probs_x))

    @poutine.scale(scale=scale)
    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        for i in pyro.irange("data", num_masked):
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs_x))
            y = pyro.sample("y_{}".format(i), dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel"})
            pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=data[i])

    @poutine.scale(scale=scale)
    @config_enumerate(default="parallel")
    def hand_guide(data):
        probs_x = pyro.param("guide_probs_x")
        for i in pyro.irange("data", num_masked):
            pyro.sample("x_{}".format(i), dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = TraceEnum_ELBO(max_iarange_nesting=1, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, auto_guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, hand_guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
@pytest.mark.parametrize('outer_obs,inner_obs',
                         [(False, True), (True, False), (True, True)])
def test_elbo_enumerate_iarange_4(outer_obs, inner_obs, scale):
    #    a ---> outer_obs
    #      \
    #  +-----\------------------+
    #  |       \                |
    #  | b ---> inner_obs   N=2 |
    #  +------------------------+
    # This tests two different observations, one outside and one inside an iarange.
    pyro.param("probs_a", torch.tensor([0.4, 0.6]), constraint=constraints.simplex)
    pyro.param("probs_b", torch.tensor([0.6, 0.4]), constraint=constraints.simplex)
    pyro.param("locs", torch.tensor([-1., 1.]))
    pyro.param("scales", torch.tensor([1., 2.]), constraint=constraints.positive)
    outer_data = torch.tensor(2.0)
    inner_data = torch.tensor([0.5, 1.5])

    @poutine.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        locs = pyro.param("locs")
        scales = pyro.param("scales")
        a = pyro.sample("a", dist.Categorical(probs_a),
                        infer={"enumerate": "parallel"})
        if outer_obs:
            pyro.sample("outer_obs", dist.Normal(0., scales[a]),
                        obs=outer_data)
        with pyro.iarange("inner", 2):
            b = pyro.sample("b", dist.Categorical(probs_b),
                            infer={"enumerate": "parallel"})
            if inner_obs:
                pyro.sample("inner_obs", dist.Normal(locs[b], scales[a]),
                            obs=inner_data)

    @poutine.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        locs = pyro.param("locs")
        scales = pyro.param("scales")
        a = pyro.sample("a", dist.Categorical(probs_a),
                        infer={"enumerate": "parallel"})
        if outer_obs:
            pyro.sample("outer_obs", dist.Normal(0., scales[a]),
                        obs=outer_data)
        for i in pyro.irange("inner", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b),
                            infer={"enumerate": "parallel"})
            if inner_obs:
                pyro.sample("inner_obs_{}".format(i), dist.Normal(locs[b], scales[a]),
                            obs=inner_data[i])

    def guide():
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


def test_elbo_enumerate_iarange_5():
    #        Guide   Model
    #                  a
    #  +---------------|--+
    #  | M=2           V  |
    #  |       b ----> c  |
    #  +------------------+
    pyro.param("model_probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("model_probs_b",
               torch.tensor([0.6, 0.4]),
               constraint=constraints.simplex)
    pyro.param("model_probs_c",
               torch.tensor([[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]],
                             [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]),
               constraint=constraints.simplex)
    pyro.param("guide_probs_b",
               torch.tensor([0.8, 0.2]),
               constraint=constraints.simplex)
    data = torch.tensor([1, 2])
    c_ind = torch.arange(3, dtype=torch.long)

    @config_enumerate(default="parallel")
    def model_iarange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.iarange("b_axis", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            pyro.sample("c",
                        dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                        obs=data)

    @config_enumerate(default="parallel")
    def guide_iarange():
        probs_b = pyro.param("guide_probs_b")
        with pyro.iarange("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b))

    @config_enumerate(default="parallel")
    def model_irange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.irange("b_axis", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b))
            pyro.sample("c_{}".format(i),
                        dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                        obs=data[i])

    @config_enumerate(default="parallel")
    def guide_irange():
        probs_b = pyro.param("guide_probs_b")
        for i in pyro.irange("b_axis", 2):
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    expected_loss = elbo.differentiable_loss(model_irange, guide_irange)
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    with pytest.raises(ValueError, match="Expected model enumeration to be no more global than guide"):
        actual_loss = elbo.differentiable_loss(model_iarange, guide_iarange)
        # This never gets run because we don't support this yet.
        _check_loss_and_grads(expected_loss, actual_loss)


@pytest.mark.parametrize('enumerate1', ['parallel', 'sequential'])
def test_elbo_enumerate_iarange_6(enumerate1):
    #     Guide           Model
    #           +-------+
    #       b ----> c <---- a
    #           |  M=2  |
    #           +-------+
    # This tests that sequential enumeration over b works, even though
    # model-side enumeration moves c into b's iarange via contraction.
    pyro.param("model_probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("model_probs_b",
               torch.tensor([0.6, 0.4]),
               constraint=constraints.simplex)
    pyro.param("model_probs_c",
               torch.tensor([[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]],
                             [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]),
               constraint=constraints.simplex)
    pyro.param("guide_probs_b",
               torch.tensor([0.8, 0.2]),
               constraint=constraints.simplex)
    data = torch.tensor([1, 2])
    c_ind = torch.arange(3, dtype=torch.long)

    @config_enumerate(default="parallel")
    def model_iarange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b))
        with pyro.iarange("b_axis", 2):
            pyro.sample("c",
                        dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                        obs=data)

    @config_enumerate(default="parallel")
    def model_irange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b))
        for i in pyro.irange("b_axis", 2):
            pyro.sample("c_{}".format(i),
                        dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                        obs=data[i])

    @config_enumerate(default=enumerate1)
    def guide():
        probs_b = pyro.param("guide_probs_b")
        pyro.sample("b", dist.Categorical(probs_b))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    expected_loss = elbo.differentiable_loss(model_irange, guide)
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    actual_loss = elbo.differentiable_loss(model_iarange, guide)
    _check_loss_and_grads(expected_loss, actual_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_iarange_7(scale):
    #  Guide    Model
    #    a -----> b
    #    |        |
    #  +-|--------|----------------+
    #  | V        V                |
    #  | c -----> d -----> e   N=2 |
    #  +---------------------------+
    # This tests a mixture of model and guide enumeration.
    pyro.param("model_probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("model_probs_b",
               torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_c",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_d",
               torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]),
               constraint=constraints.simplex)
    pyro.param("model_probs_e",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("guide_probs_a",
               torch.tensor([0.35, 0.64]),
               constraint=constraints.simplex)
    pyro.param("guide_probs_c",
               torch.tensor([[0., 1.], [1., 0.]]),  # deterministic
               constraint=constraints.simplex)
    d_ind = torch.arange(2, dtype=torch.long)

    @poutine.scale(scale=scale)
    def auto_model(data):
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        probs_e = pyro.param("model_probs_e")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b[a]),
                        infer={"enumerate": "parallel"})
        with pyro.iarange("data", 2):
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
            d = pyro.sample("d",
                            dist.Categorical(probs_d[b.unsqueeze(-1), c.unsqueeze(-1), d_ind]),
                            infer={"enumerate": "parallel"})
            pyro.sample("obs", dist.Categorical(probs_e[d]), obs=data)

    @poutine.scale(scale=scale)
    def auto_guide(data):
        probs_a = pyro.param("guide_probs_a")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a),
                        infer={"enumerate": "parallel"})
        with pyro.iarange("data", 2):
            pyro.sample("c", dist.Categorical(probs_c[a]))

    @poutine.scale(scale=scale)
    def hand_model(data):
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        probs_e = pyro.param("model_probs_e")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b[a]),
                        infer={"enumerate": "parallel"})
        for i in pyro.irange("data", 2):
            c = pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))
            d = pyro.sample("d_{}".format(i),
                            dist.Categorical(probs_d[b.unsqueeze(-1), c.unsqueeze(-1), d_ind]),
                            infer={"enumerate": "parallel"})
            pyro.sample("obs_{}".format(i), dist.Categorical(probs_e[d]), obs=data[i])

    @poutine.scale(scale=scale)
    def hand_guide(data):
        probs_a = pyro.param("guide_probs_a")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a),
                        infer={"enumerate": "parallel"})
        for i in pyro.irange("data", 2):
            pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))

    data = torch.tensor([0, 0])
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, auto_guide, data)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, hand_guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_iaranges_1(scale):
    #  +-----------------+
    #  | a ----> b   M=2 |
    #  +-----------------+
    #  +-----------------+
    #  | c ----> d   N=3 |
    #  +-----------------+
    # This tests two unrelated iaranges.
    # Each should remain uncontracted.
    pyro.param("probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("probs_b",
               torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
               constraint=constraints.simplex)
    pyro.param("probs_c",
               torch.tensor([0.75, 0.25]),
               constraint=constraints.simplex)
    pyro.param("probs_d",
               torch.tensor([[0.4, 0.6], [0.3, 0.7]]),
               constraint=constraints.simplex)
    b_data = torch.tensor([0, 1])
    d_data = torch.tensor([0, 0, 1])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        with pyro.iarange("a_axis", 2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            pyro.sample("b", dist.Categorical(probs_b[a]), obs=b_data)
        with pyro.iarange("c_axis", 3):
            c = pyro.sample("c", dist.Categorical(probs_c))
            pyro.sample("d", dist.Categorical(probs_d[c]), obs=d_data)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        for i in pyro.irange("a_axis", 2):
            a = pyro.sample("a_{}".format(i), dist.Categorical(probs_a))
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]), obs=b_data[i])
        for j in pyro.irange("c_axis", 3):
            c = pyro.sample("c_{}".format(j), dist.Categorical(probs_c))
            pyro.sample("d_{}".format(j), dist.Categorical(probs_d[c]), obs=d_data[j])

    def guide():
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_iaranges_2(scale):
    #  +---------+       +---------+
    #  |     b <---- a ----> c     |
    #  | M=2     |       |     N=3 |
    #  +---------+       +---------+
    # This tests two different iaranges with recycled dimension.
    pyro.param("probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("probs_b",
               torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
               constraint=constraints.simplex)
    pyro.param("probs_c",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    b_data = torch.tensor([0, 1])
    c_data = torch.tensor([0, 0, 1])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.iarange("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b[a]),
                        obs=b_data)
        with pyro.iarange("c_axis", 3):
            pyro.sample("c", dist.Categorical(probs_c[a]),
                        obs=c_data)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.irange("b_axis", 2):
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]),
                        obs=b_data[i])
        for j in pyro.irange("c_axis", 3):
            pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a]),
                        obs=c_data[j])

    def guide():
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_iaranges_3(scale):
    #      +--------------------+
    #      |  +----------+      |
    #  a -------> b      |      |
    #      |  |      N=2 |      |
    #      |  +----------+  M=2 |
    #      +--------------------+
    # This is tests the case of multiple iarange contractions in
    # a single step.
    pyro.param("probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("probs_b",
               torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
               constraint=constraints.simplex)
    data = torch.tensor([[0, 1], [0, 0]])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.iarange("outer", 2):
            with pyro.iarange("inner", 2):
                pyro.sample("b", dist.Categorical(probs_b[a]),
                            obs=data)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        inner = pyro.irange("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.irange("outer", 2):
            for j in inner:
                pyro.sample("b_{}_{}".format(i, j), dist.Categorical(probs_b[a]),
                            obs=data[i, j])

    def guide():
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_iaranges_4(scale):
    #      +--------------------+
    #      |       +----------+ |
    #  a ----> b ----> c      | |
    #      |       |      N=2 | |
    #      | M=2   +----------+ |
    #      +--------------------+
    pyro.param("probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("probs_b",
               torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
               constraint=constraints.simplex)
    pyro.param("probs_c",
               torch.tensor([[0.4, 0.6], [0.3, 0.7]]),
               constraint=constraints.simplex)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def auto_model(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.iarange("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.iarange("inner", 2):
                pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def hand_model(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        inner = pyro.irange("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.irange("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]))
            for j in inner:
                pyro.sample("c_{}_{}".format(i, j), dist.Categorical(probs_c[b]),
                            obs=data[i, j])

    def guide(data):
        pass

    data = torch.tensor([[0, 1], [0, 0]])
    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_iaranges_5(scale):
    #     a
    #     | \
    #  +--|---\------------+
    #  |  V   +-\--------+ |
    #  |  b ----> c      | |
    #  |      |      N=2 | |
    #  | M=2  +----------+ |
    #  +-------------------+
    pyro.param("probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("probs_b",
               torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
               constraint=constraints.simplex)
    pyro.param("probs_c",
               torch.tensor([[[0.4, 0.6], [0.3, 0.7]],
                             [[0.2, 0.8], [0.1, 0.9]]]),
               constraint=constraints.simplex)
    data = torch.tensor([[0, 1], [0, 0]])
    c_ind = torch.arange(2, dtype=torch.long)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.iarange("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.iarange("inner", 2):
                pyro.sample("c",
                            dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                            obs=data)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        inner = pyro.irange("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.irange("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]))
            for j in inner:
                pyro.sample("c_{}_{}".format(i, j),
                            dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                            obs=data[i, j])

    def guide():
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('scale', [1, 10])
def test_elbo_enumerate_iaranges_6(scale):
    #         +----------+
    #         |      M=2 |
    #     a ----> b      |
    #     |   |   |      |
    #  +--|-------|--+   |
    #  |  V   |   V  |   |
    #  |  c ----> d  |   |
    #  |      |      |   |
    #  | N=2  +------|---+
    #  +-------------+
    # This tests different ways of mixing two independence contexts,
    # where each can be either irange or iarange.
    pyro.param("probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("probs_b",
               torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
               constraint=constraints.simplex)
    pyro.param("probs_c",
               torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
               constraint=constraints.simplex)
    pyro.param("probs_d",
               torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]),
               constraint=constraints.simplex)
    d_ind = torch.arange(2, dtype=torch.long)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def model_irange_irange(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.irange("b_axis", 2)
        c_axis = pyro.irange("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a])) for i in b_axis]
        c = [pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis]
        for i in b_axis:
            for j in c_axis:
                pyro.sample("d_{}_{}".format(i, j),
                            dist.Categorical(probs_d[b[i].unsqueeze(-1), c[j].unsqueeze(-1), d_ind]),
                            obs=data[i, j])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def model_irange_iarange(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.irange("b_axis", 2)
        c_axis = pyro.iarange("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a])) for i in b_axis]
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        for i in b_axis:
            with c_axis:
                pyro.sample("d_{}".format(i),
                            dist.Categorical(probs_d[b[i].unsqueeze(-1), c.unsqueeze(-1), d_ind]),
                            obs=data[i])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def model_iarange_irange(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.iarange("b_axis", 2)
        c_axis = pyro.irange("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        c = [pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis]
        with b_axis:
            for j in c_axis:
                pyro.sample("d_{}".format(j),
                            dist.Categorical(probs_d[b.unsqueeze(-1), c[j].unsqueeze(-1), d_ind]),
                            obs=data[:, j])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=scale)
    def model_iarange_iarange(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.iarange("b_axis", 2, dim=-1)
        c_axis = pyro.iarange("c_axis", 2, dim=-2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            pyro.sample("d",
                        dist.Categorical(probs_d[b.unsqueeze(-1), c.unsqueeze(-1), d_ind]),
                        obs=data)

    def guide(data):
        pass

    # Check that either one of the iranges can be promoted to an iarange.
    data = torch.tensor([[0, 1], [0, 0]])
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    loss_irange_irange = elbo.differentiable_loss(model_irange_irange, guide, data)
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    loss_iarange_irange = elbo.differentiable_loss(model_iarange_irange, guide, data)
    loss_irange_iarange = elbo.differentiable_loss(model_irange_iarange, guide, data)
    _check_loss_and_grads(loss_irange_irange, loss_iarange_irange)
    _check_loss_and_grads(loss_irange_irange, loss_irange_iarange)

    # But promoting both to iaranges should result in an error.
    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    with pytest.raises(ValueError, match="Expected tree-structured iarange nesting.*"):
        elbo.differentiable_loss(model_iarange_iarange, guide, data)


@pytest.mark.parametrize('guide_scale', [1])
@pytest.mark.parametrize('model_scale', [1])
@pytest.mark.parametrize('outer_vectorized,inner_vectorized,xfail',
                         [(False, True, False), (True, False, True), (True, True, True)],
                         ids=['irange-iarange', 'iarange-irange', 'iarange-iarange'])
def test_elbo_enumerate_iaranges_7(model_scale, guide_scale, inner_vectorized, outer_vectorized, xfail):
    #        Guide   Model
    #                  a
    #      +-----------|--------+
    #      | M=2   +---|------+ |
    #      |       |   V  N=2 | |
    #      |   b ----> c      | |
    #      |       +----------+ |
    #      +--------------------+
    pyro.param("model_probs_a",
               torch.tensor([0.45, 0.55]),
               constraint=constraints.simplex)
    pyro.param("model_probs_b",
               torch.tensor([0.6, 0.4]),
               constraint=constraints.simplex)
    pyro.param("model_probs_c",
               torch.tensor([[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]],
                             [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]),
               constraint=constraints.simplex)
    pyro.param("guide_probs_b",
               torch.tensor([0.8, 0.2]),
               constraint=constraints.simplex)
    data = torch.tensor([[0, 1], [0, 2]])
    c_ind = torch.arange(3, dtype=torch.long)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=model_scale)
    def model_iarange_iarange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.iarange("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            with pyro.iarange("inner", 2):
                pyro.sample("c",
                            dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                            obs=data)

    @config_enumerate(default="parallel")
    @poutine.scale(scale=model_scale)
    def model_irange_iarange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        inner = pyro.iarange("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.irange("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b))
            with inner:
                pyro.sample("c_{}".format(i),
                            dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                            obs=data[:, i])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=model_scale)
    def model_iarange_irange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.iarange("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            for j in pyro.irange("inner", 2):
                pyro.sample("c_{}".format(j),
                            dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                            obs=data[j])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=model_scale)
    def model_irange_irange():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        inner = pyro.irange("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.irange("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b))
            for j in inner:
                pyro.sample("c_{}_{}".format(i, j),
                            dist.Categorical(probs_c[a.unsqueeze(-1), b.unsqueeze(-1), c_ind]),
                            obs=data[j, i])

    @config_enumerate(default="parallel")
    @poutine.scale(scale=guide_scale)
    def guide_iarange():
        probs_b = pyro.param("guide_probs_b")
        with pyro.iarange("outer", 2):
            pyro.sample("b", dist.Categorical(probs_b))

    @config_enumerate(default="parallel")
    @poutine.scale(scale=guide_scale)
    def guide_irange():
        probs_b = pyro.param("guide_probs_b")
        for i in pyro.irange("outer", 2):
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    expected_loss = elbo.differentiable_loss(model_irange_irange, guide_irange)
    with ExitStack() as stack:
        if xfail:
            stack.enter_context(pytest.raises(
                ValueError,
                match="Expected model enumeration to be no more global than guide"))
        if inner_vectorized:
            if outer_vectorized:
                elbo = TraceEnum_ELBO(max_iarange_nesting=2)
                actual_loss = elbo.differentiable_loss(model_iarange_iarange, guide_iarange)
            else:
                elbo = TraceEnum_ELBO(max_iarange_nesting=1)
                actual_loss = elbo.differentiable_loss(model_irange_iarange, guide_irange)
        else:
            elbo = TraceEnum_ELBO(max_iarange_nesting=1)
            actual_loss = elbo.differentiable_loss(model_iarange_irange, guide_iarange)
        _check_loss_and_grads(expected_loss, actual_loss)


def test_elbo_scale():
    # Consider a mixture model with two components, toggled by `which`.
    def component_model(data, which, suffix=""):
        loc = pyro.param("locs", torch.tensor([-1., 1.]))[which]
        with pyro.iarange("data" + suffix, len(data)):
            pyro.sample("obs" + suffix, dist.Normal(loc, 1.), obs=data)

    pyro.param("mixture_probs", torch.tensor([0.25, 0.75]), constraint=constraints.simplex)

    # We can implement this in two ways.
    # First consider automatic enumeration in the guide.
    def auto_model(data):
        mixture_probs = pyro.param("mixture_probs")
        which = pyro.sample("which", dist.Categorical(mixture_probs))
        component_model(data, which)

    def auto_guide(data):
        mixture_probs = pyro.param("mixture_probs")
        pyro.sample("which", dist.Categorical(mixture_probs),
                    infer={"enumerate": "parallel"})

    # Second consider explicit enumeration in the model, where we
    # marginalize out the `which` variable by hand.
    def hand_model(data):
        mixture_probs = pyro.param("mixture_probs")
        for which in pyro.irange("which", len(mixture_probs)):
            with pyro.poutine.scale(scale=mixture_probs[which]):
                component_model(data, which, suffix="_{}".format(which))

    def hand_guide(data):
        pass

    data = dist.Normal(0., 2.).sample((3,))
    elbo = TraceEnum_ELBO(max_iarange_nesting=1, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, auto_guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, hand_guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


def test_elbo_hmm_growth():
    pyro.clear_param_store()
    init_probs = torch.tensor([0.5, 0.5])
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)

    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        emission_probs = pyro.param("emission_probs",
                                    torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                    constraint=constraints.simplex)
        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))
            pyro.sample("y_{}".format(i), dist.Categorical(emission_probs[x]), obs=y)

    @config_enumerate(default="parallel")
    def guide(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))

    sizes = range(2, 16)
    costs = []
    times1 = []
    times2 = []
    for size in sizes:
        data = torch.ones(size)

        time0 = timeit.default_timer()
        elbo.loss_and_grads(model, guide, data)  # compiles paths
        time1 = timeit.default_timer()
        elbo.loss_and_grads(model, guide, data)  # reuses compiled path
        time2 = timeit.default_timer()

        times1.append(time1 - time0)
        times2.append(time2 - time1)
        costs.append(LAST_CACHE_SIZE[0])

    collated_costs = defaultdict(list)
    for counts in costs:
        for key, cost in counts.items():
            collated_costs[key].append(cost)
    print('Growth:')
    print('sizes = {}'.format(repr(sizes)))
    print('costs = {}'.format(repr(dict(collated_costs))))
    print('times1 = {}'.format(repr(times1)))
    print('times2 = {}'.format(repr(times2)))

    for key, cost in collated_costs.items():
        dt = 3
        assert cost[-1 - dt - dt] - 2 * cost[-1 - dt] + cost[-1] == 0, '{} cost is not linear'.format(key)


@pytest.mark.xfail(reason="flakey on travis due to nondeterministic computations")
@pytest.mark.skipif("CUDA_TEST" in os.environ, reason="https://github.com/uber/pyro/issues/1380")
def test_elbo_dbn_growth():
    pyro.clear_param_store()
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)

    def model(data):
        uniform = torch.tensor([0.5, 0.5])
        probs_z = pyro.param("probs_z",
                             torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                             constraint=constraints.simplex)
        for i, z in enumerate(data):
            pyro.sample("x_{}".format(i), dist.Categorical(uniform))
            y = pyro.sample("y_{}".format(i), dist.Categorical(uniform))
            pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=z)

    @config_enumerate(default="parallel")
    def guide(data):
        probs_x = pyro.param("probs_x",
                             torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y",
                             torch.tensor([[[0.75, 0.25], [0.45, 0.55]],
                                           [[0.55, 0.45], [0.25, 0.75]]]),
                             constraint=constraints.simplex)
        x = 0
        y = 0
        for i in range(len(data)):
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs_x[x]))
            y = pyro.sample("y_{}".format(i), dist.Categorical(probs_y[x, y]))

    sizes = range(2, 16)
    costs = []
    times1 = []
    times2 = []
    for size in sizes:
        data = torch.ones(size)

        time0 = timeit.default_timer()
        elbo.loss_and_grads(model, guide, data)  # compiles paths
        time1 = timeit.default_timer()
        elbo.loss_and_grads(model, guide, data)  # reuses compiled path
        time2 = timeit.default_timer()

        times1.append(time1 - time0)
        times2.append(time2 - time1)
        costs.append(LAST_CACHE_SIZE[0])

    collated_costs = defaultdict(list)
    for counts in costs:
        for key, cost in counts.items():
            collated_costs[key].append(cost)
    print('Growth:')
    print('sizes = {}'.format(repr(sizes)))
    print('costs = {}'.format(repr(dict(collated_costs))))
    print('times1 = {}'.format(repr(times1)))
    print('times2 = {}'.format(repr(times2)))

    for key, cost in collated_costs.items():
        dt = 4
        assert cost[-1 - dt - dt] - 2 * cost[-1 - dt] + cost[-1] <= 2, \
            '{} cost is not approximately linear'.format(key)


@pytest.mark.parametrize("pi_a", [0.33])
@pytest.mark.parametrize("pi_b", [0.51, 0.77])
@pytest.mark.parametrize("pi_c", [0.37])
@pytest.mark.parametrize("N_b", [3, 4])
@pytest.mark.parametrize("N_c", [5, 6])
@pytest.mark.parametrize("enumerate1", ["sequential", "parallel"])
@pytest.mark.parametrize("expand", [True, False])
def test_bernoulli_pyramid_elbo_gradient(enumerate1, N_b, N_c, pi_a, pi_b, pi_c, expand):
    pyro.clear_param_store()

    def model():
        a = pyro.sample("a", dist.Bernoulli(0.33))
        with pyro.iarange("b_iarange", N_b):
            b = pyro.sample("b", dist.Bernoulli(0.25 * a + 0.50))
            with pyro.iarange("c_iarange", N_c):
                pyro.sample("c", dist.Bernoulli(0.15 * a + 0.20 * b + 0.32))

    def guide():
        qa = pyro.param("qa", torch.tensor(pi_a, requires_grad=True))
        qb = pyro.param("qb", torch.tensor(pi_b, requires_grad=True))
        qc = pyro.param("qc", torch.tensor(pi_c, requires_grad=True))
        pyro.sample("a", dist.Bernoulli(qa))
        with pyro.iarange("b_iarange", N_b):
            pyro.sample("b", dist.Bernoulli(qb).expand_by([N_b]))
            with pyro.iarange("c_iarange", N_c):
                pyro.sample("c", dist.Bernoulli(qc).expand_by([N_c, N_b]))

    logger.info("Computing gradients using surrogate loss")
    elbo = TraceEnum_ELBO(max_iarange_nesting=2,
                          strict_enumeration_warning=True)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enumerate1, expand=expand))
    actual_grad_qa = pyro.param('qa').grad
    actual_grad_qb = pyro.param('qb').grad
    actual_grad_qc = pyro.param('qc').grad

    logger.info("Computing analytic gradients")
    qa = torch.tensor(pi_a, requires_grad=True)
    qb = torch.tensor(pi_b, requires_grad=True)
    qc = torch.tensor(pi_c, requires_grad=True)
    elbo = kl_divergence(dist.Bernoulli(qa), dist.Bernoulli(0.33))
    elbo = elbo + N_b * qa * kl_divergence(dist.Bernoulli(qb), dist.Bernoulli(0.75))
    elbo = elbo + N_b * (1.0 - qa) * kl_divergence(dist.Bernoulli(qb), dist.Bernoulli(0.50))
    elbo = elbo + N_c * N_b * qa * qb * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.67))
    elbo = elbo + N_c * N_b * (1.0 - qa) * qb * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.52))
    elbo = elbo + N_c * N_b * qa * (1.0 - qb) * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.47))
    elbo = elbo + N_c * N_b * (1.0 - qa) * (1.0 - qb) * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.32))
    expected_grad_qa, expected_grad_qb, expected_grad_qc = grad(elbo, [qa, qb, qc])

    prec = 0.001

    assert_equal(actual_grad_qa, expected_grad_qa, prec=prec, msg="".join([
        "\nqa expected = {}".format(expected_grad_qa.data.cpu().numpy()),
        "\nqa  actual = {}".format(actual_grad_qa.data.cpu().numpy()),
    ]))
    assert_equal(actual_grad_qb, expected_grad_qb, prec=prec, msg="".join([
        "\nqb expected = {}".format(expected_grad_qb.data.cpu().numpy()),
        "\nqb   actual = {}".format(actual_grad_qb.data.cpu().numpy()),
    ]))
    assert_equal(actual_grad_qc, expected_grad_qc, prec=prec, msg="".join([
        "\nqc expected = {}".format(expected_grad_qc.data.cpu().numpy()),
        "\nqc   actual = {}".format(actual_grad_qc.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("pi_a", [0.33])
@pytest.mark.parametrize("pi_b", [0.51])
@pytest.mark.parametrize("pi_c", [0.37])
@pytest.mark.parametrize("pi_d", [0.29])
@pytest.mark.parametrize("b_factor", [0.03, 0.04])
@pytest.mark.parametrize("c_factor", [0.04, 0.06])
@pytest.mark.parametrize("d_offset", [0.32])
@pytest.mark.parametrize("enumerate1", ["sequential", "parallel"])
@pytest.mark.parametrize("expand", [True, False])
def test_bernoulli_non_tree_elbo_gradient(enumerate1, b_factor, c_factor, pi_a, pi_b, pi_c, pi_d,
                                          expand, d_offset, N_b=2, N_c=2):
    pyro.clear_param_store()

    def model():
        a = pyro.sample("a", dist.Bernoulli(0.33))
        b = pyro.sample("b", dist.Bernoulli(0.25 * a + 0.50))
        c = pyro.sample("c", dist.Bernoulli(0.25 * a + 0.10 * b + 0.50))
        pyro.sample("d", dist.Bernoulli(b_factor * b + c_factor * c + d_offset))

    def guide():
        qa = pyro.param("qa", torch.tensor(pi_a, requires_grad=True))
        qb = pyro.param("qb", torch.tensor(pi_b, requires_grad=True))
        qc = pyro.param("qc", torch.tensor(pi_c, requires_grad=True))
        qd = pyro.param("qd", torch.tensor(pi_d, requires_grad=True))
        pyro.sample("a", dist.Bernoulli(qa))
        pyro.sample("b", dist.Bernoulli(qb))
        pyro.sample("c", dist.Bernoulli(qc))
        pyro.sample("d", dist.Bernoulli(qd))

    logger.info("Computing gradients using surrogate loss")
    elbo = TraceEnum_ELBO(max_iarange_nesting=2,
                          strict_enumeration_warning=True)
    elbo.loss_and_grads(model, config_enumerate(guide, default=enumerate1, expand=expand))
    actual_grad_qa = pyro.param('qa').grad
    actual_grad_qb = pyro.param('qb').grad
    actual_grad_qc = pyro.param('qc').grad
    actual_grad_qd = pyro.param('qd').grad

    logger.info("Computing analytic gradients")
    qa = torch.tensor(pi_a, requires_grad=True)
    qb = torch.tensor(pi_b, requires_grad=True)
    qc = torch.tensor(pi_c, requires_grad=True)
    qd = torch.tensor(pi_d, requires_grad=True)

    elbo = kl_divergence(dist.Bernoulli(qa), dist.Bernoulli(0.33))
    elbo = elbo + qa * kl_divergence(dist.Bernoulli(qb), dist.Bernoulli(0.75))
    elbo = elbo + (1.0 - qa) * kl_divergence(dist.Bernoulli(qb), dist.Bernoulli(0.50))

    elbo = elbo + qa * qb * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.85))
    elbo = elbo + (1.0 - qa) * qb * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.60))
    elbo = elbo + qa * (1.0 - qb) * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.75))
    elbo = elbo + (1.0 - qa) * (1.0 - qb) * kl_divergence(dist.Bernoulli(qc), dist.Bernoulli(0.50))

    elbo = elbo + qb * qc * kl_divergence(dist.Bernoulli(qd), dist.Bernoulli(b_factor + c_factor + d_offset))
    elbo = elbo + (1.0 - qb) * qc * kl_divergence(dist.Bernoulli(qd), dist.Bernoulli(c_factor + d_offset))
    elbo = elbo + qb * (1.0 - qc) * kl_divergence(dist.Bernoulli(qd), dist.Bernoulli(b_factor + d_offset))
    elbo = elbo + (1.0 - qb) * (1.0 - qc) * kl_divergence(dist.Bernoulli(qd), dist.Bernoulli(d_offset))

    expected_grad_qa, expected_grad_qb, expected_grad_qc, expected_grad_qd = grad(elbo, [qa, qb, qc, qd])

    prec = 0.0001

    assert_equal(actual_grad_qa, expected_grad_qa, prec=prec, msg="".join([
        "\nqa expected = {}".format(expected_grad_qa.data.cpu().numpy()),
        "\nqa  actual = {}".format(actual_grad_qa.data.cpu().numpy()),
    ]))
    assert_equal(actual_grad_qb, expected_grad_qb, prec=prec, msg="".join([
        "\nqb expected = {}".format(expected_grad_qb.data.cpu().numpy()),
        "\nqb   actual = {}".format(actual_grad_qb.data.cpu().numpy()),
    ]))
    assert_equal(actual_grad_qc, expected_grad_qc, prec=prec, msg="".join([
        "\nqc expected = {}".format(expected_grad_qc.data.cpu().numpy()),
        "\nqc   actual = {}".format(actual_grad_qc.data.cpu().numpy()),
    ]))
    assert_equal(actual_grad_qd, expected_grad_qd, prec=prec, msg="".join([
        "\nqd expected = {}".format(expected_grad_qd.data.cpu().numpy()),
        "\nqd   actual = {}".format(actual_grad_qd.data.cpu().numpy()),
    ]))


@pytest.mark.parametrize("gate", [0.1, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("rate", [0.1, 1., 3.])
def test_elbo_zip(gate, rate):
    # test for ZIP distribution
    @poutine.broadcast
    def zip_model(data):
        gate = pyro.param("gate")
        rate = pyro.param("rate")
        with pyro.iarange("data", len(data)):
            pyro.sample("obs", dist.ZeroInflatedPoisson(gate, rate), obs=data)

    @poutine.broadcast
    def composite_model(data):
        gate = pyro.param("gate")
        rate = pyro.param("rate")
        dist1 = dist.Delta(torch.tensor(0.))
        dist0 = dist.Poisson(rate)
        with pyro.iarange("data", len(data)):
            mask = pyro.sample("mask", dist.Bernoulli(gate), infer={"enumerate": "parallel"}).byte()
            pyro.sample("obs", dist.MaskedMixture(mask, dist0, dist1), obs=data)

    def guide(data):
        pass

    gate = pyro.param("gate", torch.tensor(gate), constraint=constraints.unit_interval)
    rate = pyro.param("rate", torch.tensor(rate), constraint=constraints.positive)

    data = torch.tensor([0., 1., 2.])
    elbo = TraceEnum_ELBO(max_iarange_nesting=1, strict_enumeration_warning=False)
    zip_loss = elbo.differentiable_loss(zip_model, guide, data)
    composite_loss = elbo.differentiable_loss(composite_model, guide, data)
    _check_loss_and_grads(zip_loss, composite_loss)


@pytest.mark.parametrize("mixture,scale", [
    (dist.MixtureOfDiagNormals, [[2., 1.], [1., 2], [4., 4.]]),
    (dist.MixtureOfDiagNormalsSharedCovariance, [2., 1.]),
])
def test_mixture_of_diag_normals(mixture, scale):
    # K = 3, D = 2
    pyro.param("locs", torch.tensor([[0., 0.], [0., 1.], [0., 10.]]))
    pyro.param("coord_scale", torch.tensor(scale), constraint=constraints.positive)
    pyro.param("component_logits", torch.tensor([0., -1., 2.]))
    data = torch.tensor([[0., 0.], [1., 1.], [2., 3.], [1., 11.]])

    def auto_model():
        locs = pyro.param("locs")
        coord_scale = pyro.param("coord_scale")
        component_logits = pyro.param("component_logits")
        with pyro.iarange("data", len(data)):
            pyro.sample("obs", mixture(locs, coord_scale, component_logits), obs=data)

    def hand_model():
        locs = pyro.param("locs")
        coord_scale = pyro.param("coord_scale")
        component_logits = pyro.param("component_logits")
        with pyro.iarange("data", len(data), dim=-2):
            which = pyro.sample("mask", dist.Categorical(logits=component_logits),
                                infer={"enumerate": "parallel"})
            with pyro.iarange("components", len(component_logits), dim=-1) as component_ind:
                with poutine.mask(mask=(which == component_ind)):
                    pyro.sample("obs", dist.Normal(locs, coord_scale).independent(1),
                                obs=data.unsqueeze(-2))

    def guide():
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=2, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("Dist, prior", [
    (dist.Bernoulli, 0.2),
    (dist.Categorical, [0.2, 0.8]),
    (dist.Categorical, [0.2, 0.3, 0.5]),
    (dist.Categorical, [0.2, 0.3, 0.3, 0.2]),
    (dist.OneHotCategorical, [0.2, 0.8]),
    (dist.OneHotCategorical, [0.2, 0.3, 0.5]),
    (dist.OneHotCategorical, [0.2, 0.3, 0.3, 0.2]),
])
def test_compute_marginals_single(Dist, prior):
    prior = torch.tensor(prior)
    data = torch.tensor([0., 0.1, 0.2, 0.9, 1.0, 1.1])

    @config_enumerate(default="parallel")
    def model():
        locs = torch.tensor([-1., 0., 1., 2.])
        x = pyro.sample("x", Dist(prior))
        if Dist is dist.Bernoulli:
            x = x.long()
        elif Dist is dist.OneHotCategorical:
            x = x.max(-1)[1]
        with pyro.iarange("data", len(data)):
            pyro.sample("obs", dist.Normal(locs[x], 1.), obs=data)

    # First compute marginals using an empty guide.
    def empty_guide():
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    marginals = elbo.compute_marginals(model, empty_guide)
    assert len(marginals) == 1
    assert type(marginals["x"]) is Dist
    probs = marginals["x"].probs
    assert probs.shape == prior.shape

    # Next insert the computed marginals in an enumerating guide
    # and ensure that they are exact, or at least locally optimal.
    pyro.param("probs", probs)

    @config_enumerate(default="parallel")
    def exact_guide():
        probs = pyro.param("probs")
        pyro.sample("x", Dist(probs))

    loss = elbo.differentiable_loss(model, exact_guide)
    assert_equal(grad(loss, [pyro.param("probs")])[0], torch.zeros_like(probs))


@pytest.mark.parametrize('ok,enumerate_guide,num_particles,vectorize_particles', [
    (True, None, 1, False),
    (False, "sequential", 1, False),
    (False, "parallel", 1, False),
    (False, None, 2, False),
    (False, None, 2, True),
])
def test_compute_marginals_restrictions(ok, enumerate_guide, num_particles, vectorize_particles):

    @config_enumerate(default="parallel")
    def model():
        w = pyro.sample("w", dist.Bernoulli(0.1))
        x = pyro.sample("x", dist.Bernoulli(0.2))
        y = pyro.sample("y", dist.Bernoulli(0.3))
        z = pyro.sample("z", dist.Bernoulli(0.4))
        pyro.sample("obs", dist.Normal(0., 1.), obs=w + x + y + z)

    @config_enumerate(default=enumerate_guide)
    def guide():
        pyro.sample("w", dist.Bernoulli(0.4))
        pyro.sample("y", dist.Bernoulli(0.7))

    # Check that the ELBO works fine.
    elbo = TraceEnum_ELBO(max_iarange_nesting=0,
                          num_particles=num_particles,
                          vectorize_particles=vectorize_particles)
    loss = elbo.loss(model, guide)
    assert not torch_isnan(loss)

    if ok:
        marginals = elbo.compute_marginals(model, guide)
        assert set(marginals.keys()) == {"x", "z"}
    else:
        with pytest.raises(NotImplementedError, match="compute_marginals"):
            elbo.compute_marginals(model, guide)


@pytest.mark.parametrize('size', [1, 2, 3, 4, 10, 20, _skip_cuda(30)])
def test_compute_marginals_hmm(size):

    @config_enumerate(default="parallel")
    def model(data):
        transition_probs = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        emission_probs = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        x = torch.tensor(0)
        for i, y in enumerate(data):
            x = pyro.sample("x_{}".format(i), dist.Categorical(transition_probs[x]))
            pyro.sample("y_{}".format(i), dist.Categorical(emission_probs[x]), obs=y)

        pyro.sample("x_{}".format(len(data)), dist.Categorical(transition_probs[x]),
                    obs=torch.tensor(1))

    def guide(data):
        pass

    data = torch.zeros(size, dtype=torch.long)
    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    marginals = elbo.compute_marginals(model, guide, data)
    assert set(marginals.keys()) == {"x_{}".format(i) for i in range(size)}
    for i in range(size):
        d = marginals["x_{}".format(i)]
        assert d.batch_shape == ()

    # The x's should be monotonically increasing, since we've observed x[-1]==0
    # and x[size]==1, and since the y's are constant.
    for i in range(size - 1):
        d1 = marginals["x_{}".format(i)]
        d2 = marginals["x_{}".format(i + 1)]
        assert d1.probs[0] > d2.probs[0]
        assert d1.probs[1] < d2.probs[1]


@pytest.mark.parametrize("data", [
    [None, None],
    [torch.tensor(0.), None],
    [None, torch.tensor(0.)],
    [torch.tensor(0.), torch.tensor(0)],
])
def test_backwardsample_posterior_smoke(data):

    @config_enumerate(default="parallel")
    def model(data):
        xs = list(data)
        zs = []
        for i in range(2):
            K = i + 2  # number of mixture components
            zs.append(pyro.sample("z_{}".format(i),
                                  dist.Categorical(torch.ones(K))))
            if i == 0:
                loc = pyro.param("loc", torch.randn(K))[zs[i]]
                xs[i] = pyro.sample("x_{}".format(i),
                                    dist.Normal(loc, 1.), obs=data[i])
            elif i == 1:
                logits = pyro.param("logits", torch.randn(K, 2))[zs[i]]
                xs[i] = pyro.sample("x_{}".format(i),
                                    dist.Categorical(logits=logits),
                                    obs=data[i])

        z12 = zs[0] + 2 * zs[1]
        pyro.sample("z_12", dist.Categorical(torch.arange(6.)), obs=z12)
        return xs, zs

    def guide(data):
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    xs, zs = elbo.sample_posterior(model, guide, data)
    for x, datum in zip(xs, data):
        assert datum is None or datum is x
    for z in zs:
        assert z.shape == ()


def test_backwardsample_posterior_2():
    num_particles = 10000

    @config_enumerate(default="parallel")
    def model(data):
        with pyro.iarange("particles", num_particles):
            p_z = torch.tensor([0.1, 0.9])
            x = pyro.sample("x", dist.Categorical(torch.tensor([0.5, 0.5])))
            z = pyro.sample("z", dist.Bernoulli(p_z[x]), obs=data)
        return x, z

    def guide(data):
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    x, z = elbo.sample_posterior(model, guide, data=torch.zeros(num_particles))
    expected = 0.9
    actual = (x.type_as(z) == z).float().mean().item()
    assert abs(expected - actual) < 0.05


def test_backwardsample_posterior_3():
    num_particles = 10000

    @config_enumerate(default="parallel")
    def model(data):
        with pyro.iarange("particles", num_particles):
            p_z = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
            x = pyro.sample("x", dist.Categorical(torch.tensor([0.5, 0.5])))
            y = pyro.sample("y", dist.Categorical(torch.tensor([0.5, 0.5])))
            z = pyro.sample("z", dist.Bernoulli(p_z[x, y]), obs=data)
        return x, y, z

    def guide(data):
        pass

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)

    x, y, z = elbo.sample_posterior(model, guide, data=torch.ones(num_particles))
    expected = 0.9
    actual = (x == y).float().mean().item()
    assert abs(expected - actual) < 0.05

    x, y, z = elbo.sample_posterior(model, guide, data=torch.zeros(num_particles))
    expected = 0.1
    actual = (x == y).float().mean().item()
    assert abs(expected - actual) < 0.05


@pytest.mark.parametrize('ok,enumerate_guide,num_particles,vectorize_particles', [
    (True, None, 1, False),
    (False, "sequential", 1, False),
    (False, "parallel", 1, False),
    (False, None, 2, False),
    (False, None, 2, True),
])
def test_backwardsample_posterior_restrictions(ok, enumerate_guide, num_particles, vectorize_particles):

    @config_enumerate(default="parallel")
    def model():
        w = pyro.sample("w", dist.Bernoulli(0.1))
        x = pyro.sample("x", dist.Bernoulli(0.2))
        y = pyro.sample("y", dist.Bernoulli(0.3))
        z = pyro.sample("z", dist.Bernoulli(0.4))
        pyro.sample("obs", dist.Normal(0., 1.), obs=w + x + y + z)
        return w, x, y, z

    @config_enumerate(default=enumerate_guide)
    def guide():
        pyro.sample("w", dist.Bernoulli(0.4))
        pyro.sample("y", dist.Bernoulli(0.7))

    # Check that the ELBO works fine.
    elbo = TraceEnum_ELBO(max_iarange_nesting=0,
                          num_particles=num_particles,
                          vectorize_particles=vectorize_particles)
    loss = elbo.loss(model, guide)
    assert not torch_isnan(loss)

    if ok:
        w, x, y, z = elbo.sample_posterior(model, guide)
        assert w.shape == ()
        assert x.shape == ()
        assert y.shape == ()
        assert z.shape == ()
    else:
        with pytest.raises(NotImplementedError, match="sample_posterior"):
            elbo.sample_posterior(model, guide)
