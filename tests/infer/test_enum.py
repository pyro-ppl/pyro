from __future__ import absolute_import, division, print_function

import logging
import math
import timeit
from collections import defaultdict

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints, kl_divergence

import pyro
import pyro.distributions as dist
import pyro.ops.einsum.shared
import pyro.optim
import pyro.poutine as poutine
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedGamma
from pyro.infer import SVI, config_enumerate
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.traceenum_elbo import TraceEnum_ELBO
from pyro.infer.util import LAST_CACHE_SIZE
from pyro.util import torch_isnan
from tests.common import assert_equal

logger = logging.getLogger(__name__)


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
    prec = 0.001 if enumerate1 else 0.1
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
        prec = 0.1

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
    ("parallel", 30, False),
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
    ("parallel", 30, False),
    ("parallel", 40, False),
    ("parallel", 50, False),
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


@pytest.mark.parametrize('num_steps', [2, 3, 4, 5, 10, 20, 30])
def test_hmm_enumerate_model(num_steps):
    data = dist.Categorical(torch.tensor([0.5, 0.5])).sample((num_steps,))

    @config_enumerate(default="parallel", expand=False)
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


@pytest.mark.parametrize('num_steps', [2, 3, 4, 5, 10, 20, 30])
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
                            infer={"enumerate": "parallel", "expand": False})
            pyro.sample("y_{}".format(t), dist.Categorical(emission_probs[x]), obs=y)
            print('{}\t{}'.format(t, tuple(x.shape)))

    def guide(data):
        init_probs = pyro.param("init_probs",
                                torch.tensor([0.75, 0.25]),
                                constraint=constraints.simplex)
        pyro.sample("x", dist.Categorical(init_probs),
                    infer={"enumerate": "parallel", "expand": False})

    elbo = TraceEnum_ELBO(max_iarange_nesting=0)
    elbo.differentiable_loss(model, guide, data)


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
                    infer={"enumerate": "parallel", "expand": False})
        pyro.sample("z", dist.Categorical(probs_z), obs=torch.tensor(0))

    @poutine.scale(scale=scale)
    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_z = pyro.param("model_probs_z")
        pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("z", dist.Categorical(probs_z), obs=torch.tensor(0))

    @config_enumerate(default="parallel", expand=False)
    @poutine.scale(scale=scale)
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    assert_equal(auto_loss, hand_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(hand_loss.detach().cpu().numpy(),
                                                         auto_loss.detach().cpu().numpy()))

    names = ["guide_probs_x", "model_probs_x", "model_probs_y"]
    params = [pyro.param(name).unconstrained() for name in names]
    auto_grads = grad(auto_loss, params, allow_unused=True)
    hand_grads = grad(hand_loss, params, allow_unused=True)
    for name, auto_grad, hand_grad in zip(names, auto_grads, hand_grads):
        if auto_grad is None or hand_grad is None:
            continue
        assert_equal(auto_grad, hand_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 hand_grad.detach().cpu().numpy(),
                                                                 auto_grad.detach().cpu().numpy()))


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

    def auto_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        pyro.sample("w", dist.Bernoulli(0.5),
                    infer={"enumerate": "parallel", "expand": False})
        x = pyro.sample("x", dist.Categorical(probs_x))
        y = pyro.sample("y", dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel", "expand": False})
        with poutine.scale(scale=scale):
            pyro.sample("z", dist.Categorical(probs_z[y]), obs=torch.tensor(0))
            print('DEBUG probs_z[y].shape = {}'.format(probs_z[y].shape))

    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        pyro.sample("w", dist.Bernoulli(0.5),
                    infer={"enumerate": "parallel", "expand": False})
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            pyro.sample("z", dist.Categorical(probs_yz[x]), obs=torch.tensor(0))
            print('DEBUG probs_yz[x].shape = {}'.format(probs_yz[x].shape))

    @config_enumerate(default="parallel", expand=False)
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    assert_equal(auto_loss, hand_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(hand_loss.detach().cpu().numpy(),
                                                         auto_loss.detach().cpu().numpy()))

    names = ["guide_probs_x", "model_probs_x", "model_probs_y", "model_probs_z"]
    params = [pyro.param(name).unconstrained() for name in names]
    auto_grads = grad(auto_loss, params, allow_unused=True)
    hand_grads = grad(hand_loss, params, allow_unused=True)
    for name, auto_grad, hand_grad in zip(names, auto_grads, hand_grads):
        assert_equal(auto_grad, hand_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 hand_grad.detach().cpu().numpy(),
                                                                 auto_grad.detach().cpu().numpy()))


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
        pyro.sample("w", dist.Bernoulli(0.5),
                    infer={"enumerate": "parallel", "expand": False})
        x = pyro.sample("x", dist.Categorical(probs_x))
        y = pyro.sample("y", dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel", "expand": False})
        with poutine.scale(scale=scale):
            pyro.sample("z1", dist.Categorical(probs_z[y]), obs=torch.tensor(0))
            pyro.sample("z2", dist.Categorical(probs_z[y]), obs=torch.tensor(0))
            print('DEBUG probs_z[y].shape = {}'.format(probs_z[y].shape))

    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        pyro.sample("w", dist.Bernoulli(0.5),
                    infer={"enumerate": "parallel", "expand": False})
        x = pyro.sample("x", dist.Categorical(probs_x))
        with poutine.scale(scale=scale):
            pyro.sample("z1", dist.Categorical(probs_yz[x]), obs=torch.tensor(0))
            pyro.sample("z2", dist.Categorical(probs_yz[x]), obs=torch.tensor(0))
            print('DEBUG probs_yz[x].shape = {}'.format(probs_yz[x].shape))

    @config_enumerate(default="parallel", expand=False)
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = TraceEnum_ELBO(max_iarange_nesting=0, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    assert_equal(auto_loss, hand_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(hand_loss.detach().cpu().numpy(),
                                                         auto_loss.detach().cpu().numpy()))

    names = ["guide_probs_x", "model_probs_x", "model_probs_y", "model_probs_z"]
    params = [pyro.param(name).unconstrained() for name in names]
    auto_grads = grad(auto_loss, params, allow_unused=True)
    hand_grads = grad(hand_loss, params, allow_unused=True)
    for name, auto_grad, hand_grad in zip(names, auto_grads, hand_grads):
        assert_equal(auto_grad, hand_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 hand_grad.detach().cpu().numpy(),
                                                                 auto_grad.detach().cpu().numpy()))


@pytest.mark.parametrize('scale', [1, 10])
@pytest.mark.parametrize('num_samples', [1, 3])
def test_elbo_enumerate_iarange_1(num_samples, scale):
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
        x = pyro.sample("x", dist.Categorical(probs_x))
        y = pyro.sample("y", dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel", "expand": False})
        with pyro.iarange("data", len(data)):
            pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    @poutine.scale(scale=scale)
    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        x = pyro.sample("x", dist.Categorical(probs_x))
        with pyro.iarange("data", len(data)):
            pyro.sample("z", dist.Categorical(probs_yz[x]), obs=data)

    @poutine.scale(scale=scale)
    @config_enumerate(default="parallel", expand=False)
    def guide(data):
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = TraceEnum_ELBO(max_iarange_nesting=1, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    assert_equal(auto_loss, hand_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(hand_loss.detach().cpu().numpy(),
                                                         auto_loss.detach().cpu().numpy()))

    names = ["guide_probs_x", "model_probs_x", "model_probs_y", "model_probs_z"]
    params = [pyro.param(name).unconstrained() for name in names]
    auto_grads = grad(auto_loss, params, allow_unused=True)
    hand_grads = grad(hand_loss, params, allow_unused=True)
    for name, auto_grad, hand_grad in zip(names, auto_grads, hand_grads):
        assert_equal(auto_grad, hand_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 hand_grad.detach().cpu().numpy(),
                                                                 auto_grad.detach().cpu().numpy()))


@pytest.mark.parametrize('scale', [1, 10])
@pytest.mark.parametrize('num_samples', [1,  3])
def test_elbo_enumerate_iarange_2(num_samples, scale):
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
        with pyro.iarange("data", len(data)):
            y = pyro.sample("y", dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel", "expand": False})
            with poutine.scale(scale=scale):
                pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
                print('DEBUG probs_z[y].shape = {}'.format(probs_z[y].shape))

    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        x = pyro.sample("x", dist.Categorical(probs_x))
        with pyro.iarange("data", len(data)):
            with poutine.scale(scale=scale):
                pyro.sample("z", dist.Categorical(probs_yz[x]), obs=data)
                print('DEBUG probs_yz[x].shape = {}'.format(probs_yz[x].shape))

    @config_enumerate(default="parallel", expand=False)
    def guide(data):
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = TraceEnum_ELBO(max_iarange_nesting=1, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    assert_equal(auto_loss, hand_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(hand_loss.detach().cpu().numpy(),
                                                         auto_loss.detach().cpu().numpy()))

    names = ["guide_probs_x", "model_probs_x", "model_probs_y", "model_probs_z"]
    params = [pyro.param(name).unconstrained() for name in names]
    auto_grads = grad(auto_loss, params, allow_unused=True)
    hand_grads = grad(hand_loss, params, allow_unused=True)
    for name, auto_grad, hand_grad in zip(names, auto_grads, hand_grads):
        assert_equal(auto_grad, hand_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 hand_grad.detach().cpu().numpy(),
                                                                 auto_grad.detach().cpu().numpy()))


@pytest.mark.parametrize('num_samples', [1,  3])
def test_elbo_enumerate_iarange_3(num_samples):
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
        with pyro.iarange("data", len(data)):
            x = pyro.sample("x", dist.Categorical(probs_x))
            y = pyro.sample("y", dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel", "expand": False})
            pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        with pyro.iarange("data", len(data)):
            x = pyro.sample("x", dist.Categorical(probs_x))
            pyro.sample("z", dist.Categorical(probs_yz[x]), obs=data)

    @config_enumerate(default="parallel", expand=False)
    def guide(data):
        probs_x = pyro.param("guide_probs_x")
        with pyro.iarange("data", len(data)):
            pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = TraceEnum_ELBO(max_iarange_nesting=1, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    assert_equal(auto_loss, hand_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(hand_loss.detach().cpu().numpy(),
                                                         auto_loss.detach().cpu().numpy()))

    names = ["guide_probs_x", "model_probs_x", "model_probs_y", "model_probs_z"]
    params = [pyro.param(name).unconstrained() for name in names]
    auto_grads = grad(auto_loss, params, allow_unused=True)
    hand_grads = grad(hand_loss, params, allow_unused=True)
    for name, auto_grad, hand_grad in zip(names, auto_grads, hand_grads):
        assert_equal(auto_grad, hand_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 hand_grad.detach().cpu().numpy(),
                                                                 auto_grad.detach().cpu().numpy()))


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

    @config_enumerate(default="parallel", expand=False)
    def guide(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
                                      constraint=constraints.simplex)
        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))

    sizes = range(2, 11)
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

    # This assertion may fail nondeterministically:
    # assert costs[-3] + costs[-1] == 2 * costs[-2], 'cost is not asymptotically linear'


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

    @config_enumerate(default="parallel", expand=False)
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

    sizes = range(2, 11)
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

    # This assertion may fail nondeterministically:
    # assert costs[-3] + costs[-1] == 2 * costs[-2], 'cost is not asymptotically linear'


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
