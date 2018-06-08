from __future__ import absolute_import, division, print_function

import logging
import math

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
def test_iter_discrete_traces_vector(graph_type):
    pyro.clear_param_store()

    @config_enumerate
    def model():
        p = pyro.param("p", torch.tensor([0.05, 0.15]))
        probs = pyro.param("probs", torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                                  [0.4, 0.3, 0.2, 0.1]]))
        with pyro.iarange("iarange", 2):
            x = pyro.sample("x", dist.Bernoulli(p))
            y = pyro.sample("y", dist.Categorical(probs))
        assert x.size() == (2,)
        assert y.size() == (2,)
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
    data = torch.arange(0, data_size)
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
        assert z.shape[-2:] == (n, 2)
        loc = (z * mus).sum(-1)
        pyro.sample("x", dist.Normal(loc, scale.expand(n)), obs=data[batch])


def gmm_batch_guide(data):
    with pyro.iarange("data", len(data)) as batch:
        n = len(batch)
        probs = pyro.param("probs", torch.tensor(torch.ones(n, 1) * 0.6, requires_grad=True))
        probs = torch.cat([probs, 1 - probs], dim=1)
        z = pyro.sample("z", dist.OneHotCategorical(probs))
        assert z.shape[-2:] == (n, 2)


@pytest.mark.parametrize("data_size", [1, 2, 3])
@pytest.mark.parametrize("graph_type", ["flat", "dense"])
@pytest.mark.parametrize("model", [gmm_batch_model, gmm_batch_guide])
def test_gmm_batch_iter_discrete_traces(model, data_size, graph_type):
    pyro.clear_param_store()
    data = torch.arange(0, data_size)
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


@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_svi_step_guide_uses_grad(enumerate1):
    data = torch.tensor([0., 1., 3.])

    @poutine.broadcast
    def model():
        scale = pyro.param("scale")
        loc = pyro.sample("loc", dist.Normal(0., 10.))
        with pyro.iarange("data", len(data)):
            pyro.sample("obs", dist.Normal(loc, scale), obs=data)
        pyro.sample("b", dist.Bernoulli(0.5))

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


@pytest.mark.parametrize("quantity", ["loss", "grad"])
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_elbo_bern(quantity, enumerate1):
    pyro.clear_param_store()
    num_particles = 1 if enumerate1 else 10000
    prec = 0.001 if enumerate1 else 0.1
    q = pyro.param("q", torch.tensor(0.5, requires_grad=True))
    kl = kl_divergence(dist.Bernoulli(q), dist.Bernoulli(0.25))

    def model():
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Bernoulli(0.25).expand_by([num_particles]))

    @config_enumerate(default=enumerate1)
    def guide():
        q = pyro.param("q")
        with pyro.iarange("particles", num_particles):
            pyro.sample("z", dist.Bernoulli(q).expand_by([num_particles]))

    elbo = TraceEnum_ELBO(max_iarange_nesting=1,
                          strict_enumeration_warning=any([enumerate1]))

    if quantity == "loss":
        actual = elbo.loss(model, guide) / num_particles
        expected = kl.item()
        assert_equal(actual, expected, prec=prec, msg="".join([
            "\nexpected = {}".format(expected),
            "\n  actual = {}".format(actual),
        ]))
    else:
        elbo.loss_and_grads(model, guide)
        actual = q.grad / num_particles
        expected = grad(kl, [q])[0]
        assert_equal(actual, expected, prec=prec, msg="".join([
            "\nexpected = {}".format(expected.detach().cpu().numpy()),
            "\n  actual = {}".format(actual.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate2", [None, "sequential", "parallel"])
@pytest.mark.parametrize("enumerate3", [None, "sequential", "parallel"])
def test_elbo_berns(enumerate1, enumerate2, enumerate3):
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
@pytest.mark.parametrize("enumerate1", [None, "sequential", "parallel"])
def test_non_mean_field_bern_normal_elbo_gradient(enumerate1, pi1, pi2, pi3, include_z=True):
    pyro.clear_param_store()
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


@pytest.mark.parametrize("enumerate1,num_steps", [
    ("sequential", 2),
    ("sequential", 3),
    ("parallel", 2),
    ("parallel", 3),
    ("parallel", 10),
    ("parallel", 20),
    pytest.param("parallel", 30, marks=pytest.mark.skip(reason="extremely expensive")),
])
def test_elbo_hmm_in_model(enumerate1, num_steps):
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

    @config_enumerate(default=enumerate1)
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


@pytest.mark.parametrize("enumerate1,num_steps", [
    ("sequential", 2),
    ("sequential", 3),
    ("parallel", 2),
    ("parallel", 3),
    ("parallel", 10),
    ("parallel", 20),
    pytest.param("parallel", 30, marks=pytest.mark.skip(reason="extremely expensive")),
])
def test_elbo_hmm_in_guide(enumerate1, num_steps):
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

    @config_enumerate(default=enumerate1)
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
    }

    for name, value in pyro.get_param_store().named_parameters():
        actual = value.grad
        expected = torch.tensor(expected_grads[num_steps][name])
        assert_equal(actual, expected, msg=''.join([
            '\nexpected {}.grad = {}'.format(name, expected.cpu().numpy()),
            '\n  actual {}.grad = {}'.format(name, actual.detach().cpu().numpy()),
        ]))
