from __future__ import absolute_import, division, print_function

import logging
import math

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import TraceEnum_ELBO
from pyro.infer.collapse import collapse
from pyro.infer.discrete import infer_discrete
from pyro.infer.enum import config_enumerate
from tests.common import assert_equal

logger = logging.getLogger(__name__)


def logsumexp(iterable):
    return sum(x.exp() for x in iterable).log()


def log_mean_prob(trace, particle_dim):
    """
    Marginalizes out particle dim from a trace.
    """
    assert particle_dim < 0
    trace.compute_log_prob()
    total = 0.
    num_particles = None
    for node in trace.nodes.values():
        if node["type"] == "sample" and type(node["fn"]).__name__ != "_Subsample":
            log_prob = node["log_prob"]
            assert log_prob.dim() == -particle_dim
            num_particles = log_prob.size(0)
            total = total + log_prob.reshape(num_particles, -1).sum(-1)
    return total.reshape(-1).logsumexp(0) - math.log(num_particles)


def test_collapse_guide_smoke():
    pyro.clear_param_store()

    def guide():
        loc_z2 = pyro.param("loc_z2", torch.randn(3))
        loc_x2 = pyro.param("loc_x2", torch.randn(3))
        scale_x3 = pyro.param("scale_x3", torch.randn(3).exp(),
                              constraint=constraints.positive)
        e = pyro.sample("e", dist.Categorical(torch.ones(3)))
        z1 = pyro.sample("z1", dist.Normal(0., 1.))
        z2 = pyro.sample("z2", dist.Normal(loc_z2[e], 1.0))
        x1 = pyro.sample("x1", dist.Normal(z2, 1.), obs=torch.tensor(1.7))
        x2 = pyro.sample("x2", dist.Normal(loc_x2[e], 1.), obs=torch.tensor(2.))
        x3 = pyro.sample("x3", dist.Normal(z1, scale_x3[e]), obs=torch.tensor(1.6))
        x4 = pyro.sample("x4", dist.Normal(z1, 1.), obs=torch.tensor(1.5))
        return x1, x2, x3, x4

    guide = poutine.infer_config(
            guide,
            lambda msg: {"enumerate": "parallel"} if msg["name"] == "e" else {})

    collapsed_guide = collapse(guide, first_available_dim=-1)

    uncollapsed_tr = poutine.trace(guide).get_trace()
    collapsed_tr = poutine.trace(collapsed_guide).get_trace()

    assert set(collapsed_tr.nodes.keys()) == set(uncollapsed_tr.nodes.keys()) - {"e"}


@pytest.mark.parametrize('temperature', [0, 1], ids=['map', 'sample'])
def test_infer_discrete_1(temperature):
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = torch.tensor([1., 2., 3.])

    @config_enumerate
    def model(num_particles=1, z=None):
        p = pyro.param("p", torch.tensor(0.25))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z = pyro.sample("z", dist.Bernoulli(p), obs=z)
            logger.info("z.shape = {}".format(z.shape))
            with pyro.plate("data", 3):
                pyro.sample("x", dist.Normal(z, 1.), obs=data)

    first_available_dim = -3
    sampled_model = infer_discrete(model, first_available_dim, temperature)
    sampled_trace = poutine.trace(sampled_model).get_trace(num_particles)
    conditioned_traces = {z: poutine.trace(model).get_trace(z=torch.tensor(z)) for z in [0., 1.]}

    # Check  posterior over z.
    actual_z_mean = sampled_trace.nodes["z"]["value"].mean()
    if temperature:
        expected_z_mean = 1 / (1 + (conditioned_traces[0].log_prob_sum() -
                                    conditioned_traces[1].log_prob_sum()).exp())
    else:
        expected_z_mean = (conditioned_traces[1].log_prob_sum() >
                           conditioned_traces[0].log_prob_sum()).float()
    assert_equal(actual_z_mean, expected_z_mean, prec=1e-2)


@pytest.mark.parametrize('temperature', [0, 1], ids=['map', 'sample'])
def test_infer_discrete_2(temperature):
    #       +--------+
    #  z1 --|--> x1  |
    #   |   |        |
    #   V   |        |
    #  z2 --|--> x2  |
    #       +--------+
    num_particles = 10000
    data = torch.tensor([[-1., -1., 0.], [-1., 1., 1.]])

    @config_enumerate
    def model(num_particles=1, z1=None, z2=None):
        p = pyro.param("p", torch.tensor([[0.25, 0.75], [0.1, 0.9]]))
        loc = pyro.param("loc", torch.tensor([-1., 1.]))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z1 = pyro.sample("z1", dist.Categorical(p[0]), obs=z1)
            z2 = pyro.sample("z2", dist.Categorical(p[z1]), obs=z2)
            logger.info("z1.shape = {}".format(z1.shape))
            logger.info("z2.shape = {}".format(z2.shape))
            with pyro.plate("data", 3):
                pyro.sample("x1", dist.Normal(loc[z1], 1.), obs=data[0])
                pyro.sample("x2", dist.Normal(loc[z2], 1.), obs=data[1])

    first_available_dim = -3
    sampled_model = infer_discrete(model, first_available_dim, temperature)
    sampled_trace = poutine.trace(
        sampled_model).get_trace(num_particles)
    conditioned_traces = {(z1, z2): poutine.trace(model).get_trace(z1=torch.tensor(z1),
                                                                   z2=torch.tensor(z2))
                          for z1 in [0, 1] for z2 in [0, 1]}

    # Check joint posterior over (z1, z2).
    actual_probs = torch.empty(2, 2)
    expected_probs = torch.empty(2, 2)
    for (z1, z2), tr in conditioned_traces.items():
        expected_probs[z1, z2] = tr.log_prob_sum().exp()
        actual_probs[z1, z2] = ((sampled_trace.nodes["z1"]["value"] == z1) &
                                (sampled_trace.nodes["z2"]["value"] == z2)).float().mean()
    if temperature:
        expected_probs = expected_probs / expected_probs.sum()
    else:
        argmax = expected_probs.reshape(-1).max(0)[1]
        expected_probs[:] = 0
        expected_probs.reshape(-1)[argmax] = 1
    assert_equal(expected_probs, actual_probs, prec=1e-2)


@pytest.mark.parametrize('temperature', [0, 1], ids=['map', 'sample'])
def test_infer_discrete_3(temperature):
    #       +---------+  +---------------+
    #  z1 --|--> x1   |  |  z2 ---> x2   |
    #       |       3 |  |             2 |
    #       +---------+  +---------------+
    num_particles = 10000
    data = [torch.tensor([-1., -1., 0.]), torch.tensor([-1., 1.])]

    @config_enumerate
    def model(num_particles=1, z1=None, z2=None):
        p = pyro.param("p", torch.tensor([0.25, 0.75]))
        loc = pyro.param("loc", torch.tensor([-1., 1.]))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z1 = pyro.sample("z1", dist.Categorical(p), obs=z1)
            with pyro.plate("data[0]", 3):
                pyro.sample("x1", dist.Normal(loc[z1], 1.), obs=data[0])
            with pyro.plate("data[1]", 2):
                z2 = pyro.sample("z2", dist.Categorical(p), obs=z2)
                pyro.sample("x2", dist.Normal(loc[z2], 1.), obs=data[1])

    first_available_dim = -3
    sampled_model = infer_discrete(model, first_available_dim, temperature)
    sampled_trace = poutine.trace(
        sampled_model).get_trace(num_particles)
    conditioned_traces = {(z1, z20, z21): poutine.trace(model).get_trace(z1=torch.tensor(z1),
                                                                         z2=torch.tensor([z20, z21]))
                          for z1 in [0, 1] for z20 in [0, 1] for z21 in [0, 1]}

    # Check joint posterior over (z1, z2[0], z2[1]).
    actual_probs = torch.empty(2, 2, 2)
    expected_probs = torch.empty(2, 2, 2)
    for (z1, z20, z21), tr in conditioned_traces.items():
        expected_probs[z1, z20, z21] = tr.log_prob_sum().exp()
        actual_probs[z1, z20, z21] = ((sampled_trace.nodes["z1"]["value"] == z1) &
                                      (sampled_trace.nodes["z2"]["value"][..., :1] == z20) &
                                      (sampled_trace.nodes["z2"]["value"][..., 1:] == z21)).float().mean()
    if temperature:
        expected_probs = expected_probs / expected_probs.sum()
    else:
        argmax = expected_probs.reshape(-1).max(0)[1]
        expected_probs[:] = 0
        expected_probs.reshape(-1)[argmax] = 1
    assert_equal(expected_probs.reshape(-1), actual_probs.reshape(-1), prec=1e-2)


@pytest.mark.xfail(reason="misunderstanding of collapse behavior")
def test_collapse_grad_1():
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = torch.tensor([1., 2., 3.])
    p = pyro.param("p", torch.tensor(0.25))

    @config_enumerate
    def model(num_particles=1, z=None):
        p = pyro.param("p")
        with pyro.plate("num_particles", num_particles, dim=-2):
            z = pyro.sample("z", dist.Bernoulli(p), obs=z)
            logger.info("z.shape = {}".format(z.shape))
            with pyro.plate("data", 3):
                pyro.sample("x", dist.Normal(z, 1.), obs=data)

    collapsed_trace = poutine.trace(collapse(model, first_available_dim=-3)).get_trace(num_particles)
    conditioned_traces = {z: poutine.trace(model).get_trace(z=torch.tensor(z)) for z in [0, 1]}

    actual_logprob = log_mean_prob(collapsed_trace, particle_dim=-2)
    expected_logprob = logsumexp(tr.log_prob_sum() for tr in conditioned_traces.values())
    assert_equal(expected_logprob, actual_logprob, prec=1e-3)

    expected_grad = grad(expected_logprob, [p], create_graph=True)[0]
    actual_grad = grad(actual_logprob, [p], create_graph=True)[0]
    assert_equal(expected_grad, actual_grad, prec=1e-3)


@pytest.mark.xfail(reason="misunderstanding of collapse behavior")
def test_collapse_grad_2():
    #       +--------+
    #  z1 --|--> x1  |
    #   |   |        |
    #   V   |        |
    #  z2 --|--> x2  |
    #       +--------+
    num_particles = 10000
    data = torch.tensor([[-1., -1., 0.], [-1., 1., 1.]])
    p = pyro.param("p", torch.tensor([[0.25, 0.75], [0.1, 0.9]]))
    loc = pyro.param("loc", torch.tensor([-1., 1.]))

    @config_enumerate
    def model():
        p = pyro.param("p")
        loc = pyro.param("loc")
        with pyro.plate("num_particles", num_particles, dim=-2):
            z1 = pyro.sample("z1", dist.Categorical(p[0]))
            z2 = pyro.sample("z2", dist.Categorical(p[z1]))
            logger.info("z1.shape = {}".format(z1.shape))
            logger.info("z2.shape = {}".format(z2.shape))
            with pyro.plate("data", 3):
                pyro.sample("x1", dist.Normal(loc[z1], 1.), obs=data[0])
                pyro.sample("x2", dist.Normal(loc[z2], 1.), obs=data[1])

    def hand_model():
        p = pyro.param("p")
        loc = pyro.param("loc")
        with pyro.plate("data", 3):
            for z1 in [0, 1]:
                for z2 in [0, 1]:
                    with poutine.scale(scale=p[0][z1] * p[z1][z2]):
                        pyro.sample("x1_{}_{}".format(z1, z2), dist.Normal(loc[z1], 1.), obs=data[0])
                        pyro.sample("x2_{}_{}".format(z1, z2), dist.Normal(loc[z2], 1.), obs=data[1])

    actual_logprob = poutine.trace(collapse(model, -3)).get_trace().log_prob_sum() / num_particles
    expected_logprob = poutine.trace(hand_model).get_trace().log_prob_sum()
    assert_equal(expected_logprob, actual_logprob, prec=1e-3)

    actual_grads = grad(actual_logprob, [p, loc], create_graph=True)
    expected_grads = grad(expected_logprob, [p, loc], create_graph=True)
    for a, e, name in zip(actual_grads, expected_grads, ["p", "loc"]):
        assert_equal(e, a, prec=1e-3, msg="bad grad for {}".format(name))


def test_collapse_traceenumelbo_smoke():
    pyro.clear_param_store()

    def guide():
        loc_z = pyro.param("loc_z_guide", torch.randn(3))
        e = pyro.sample("e", dist.Categorical(torch.ones(3)))
        pyro.sample("z", dist.Normal(loc_z[e], 1.0))

    def model():
        loc_z = pyro.param("loc_z_model", torch.randn(3))
        z = pyro.sample("z", dist.Normal(loc_z, 1.0))
        pyro.sample("x", dist.Normal(z, 1.), obs=torch.tensor(1.7))

    guide = poutine.infer_config(
        guide,
        lambda msg: {"enumerate": "parallel"} if msg["name"] == "e" else {})

    collapsed_guide = collapse(guide, first_available_dim=-1)

    elbo = TraceEnum_ELBO(max_plate_nesting=0, strict_enumeration_warning=False)
    elbo.differentiable_loss(model, collapsed_guide)


@pytest.mark.xfail(reason="We don't handle this enumeration scenario with collapse yet")
def test_collapse_elbo_categorical():

    @config_enumerate(default="parallel")
    def guide(by_hand):

        p1 = pyro.param("p1", torch.tensor([0.25, 0.75]))
        p2 = pyro.param("p2", torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]]))

        if by_hand:
            p = p2.t().mv(p1)
            z2 = pyro.sample("z2", dist.Categorical(p))
            print("uncollapsed guide z2 shape = {}".format(z2.shape))
        else:
            z1 = pyro.sample("z1", dist.Categorical(p1),
                             infer={"enumerate": "parallel"})
            print("collapsed guide z1 shape = {}".format(z1.shape))
            z2 = pyro.sample("z2", dist.Categorical(p2[z1]))
            print("collapsed guide z2 shape = {}".format(z2.shape))

    @config_enumerate(default="parallel")
    def model(by_hand):
        p = pyro.param("p2_model", torch.tensor([0.3, 0.6, 0.1]))
        locs = pyro.param("loc_x", torch.tensor([1.5, -0.8, 0.5]))
        z2 = pyro.sample("z2", dist.Categorical(p))
        print("model z2 shape = {}".format(z2.shape))
        pyro.sample("x", dist.Normal(locs[z2], 1.), obs=torch.tensor(0.))

    collapsed_guide = collapse(guide, first_available_dim=-1)

    # actual test
    pyro.infer.enable_validation(False)

    elbo = TraceEnum_ELBO(max_plate_nesting=1,  # XXX what should this be?
                          strict_enumeration_warning=False)

    expected = elbo.differentiable_loss(model, guide, True)
    actual = elbo.differentiable_loss(model, collapsed_guide, False)

    assert_equal(expected, actual)


@pytest.mark.xfail(reason="misunderstanding of collapse behavior")
def test_collapse_enum_interaction_smoke():

    # @config_enumerate(default="parallel")
    def guide(by_hand):

        p1 = pyro.param("p1", torch.tensor([0.25, 0.75]))
        p2 = pyro.param("p2", torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]]))

        if by_hand:
            p = p2.t().mv(p1)
            z2 = pyro.sample("z2", dist.Categorical(p))
            print("uncollapsed guide z2 shape = {}".format(z2.shape))
        else:
            z1 = pyro.sample("z1", dist.Categorical(p1),
                             infer={"enumerate": "parallel"})
            print("collapsed guide z1 shape = {}".format(z1.shape))
            z2 = pyro.sample("z2", dist.Categorical(p2[z1]))
            print("collapsed guide z2 shape = {}".format(z2.shape))

    @config_enumerate(default="parallel")
    def model(by_hand):
        p = pyro.param("p2_model", torch.tensor([0.3, 0.6, 0.1]))
        locs = pyro.param("loc_x", torch.tensor([1.5, -0.8, 0.5]))
        z2 = pyro.sample("z2", dist.Categorical(p))
        print("model z2 shape = {}".format(z2.shape))
        pyro.sample("x", dist.Normal(locs[z2], 1.), obs=torch.tensor(0.))

    elbo = TraceEnum_ELBO(max_plate_nesting=2,  # XXX what should this be?
                          strict_enumeration_warning=False)

    collapsed_guide = collapse(guide, first_available_dim=-1)
    # XXX not correct first_available_dim
    enum_collapsed_guide = poutine.enum(
        config_enumerate(collapsed_guide, default="parallel"),
        first_available_dim=-1)

    # smoke tests
    print("\n-------- hand collapsed, no enum")
    tr1 = poutine.trace(guide).get_trace(True)
    assert tr1.nodes["z2"]["value"].shape == ()

    print("\n-------- collapsed, no enum")
    tr2 = poutine.trace(collapsed_guide).get_trace(False)
    assert tr2.nodes["z2"]["value"].shape == () and "z1" not in tr2

    tr12 = poutine.trace(poutine.replay(guide, trace=tr2)).get_trace(True)
    assert tr12.log_prob_sum() == tr2.log_prob_sum()

    print("\n-------- collapsed, enum")
    tr3 = poutine.trace(enum_collapsed_guide).get_trace(False)
    assert tr3.nodes["z2"]["value"].shape == (3,)

    print("\n-------- collapsed, backwardsample")
    tr4 = poutine.trace(elbo.sample_posterior).get_trace(
        model, collapsed_guide, False)
    assert "z2" not in tr4.nodes
