from __future__ import absolute_import, division, print_function

import logging
import pytest

import torch
from torch.distributions import constraints
from torch.autograd import grad

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.infer.enum import config_enumerate
from pyro.infer.collapse import collapse
from tests.common import assert_equal

logger = logging.getLogger(__name__)


def test_collapse_exact_inference():
    data = torch.tensor([1., 2., 3.])
    num_particles = 10000

    @config_enumerate(default="parallel")
    def model():
        p1 = pyro.param("p1", torch.tensor(0.25))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z = pyro.sample("z", dist.Bernoulli(p1), infer={"collapse": True})
            with pyro.plate("data", 3):
                pyro.sample("x", dist.Normal(z, 1.), obs=data)

    def hand_model():
        p1 = pyro.param("p1", torch.tensor(0.25))
        with pyro.plate("data", 3):
            with poutine.scale(scale=1.-p1):
                pyro.sample("x0", dist.Normal(0., 1.), obs=data)
            with poutine.scale(scale=p1):
                pyro.sample("x1", dist.Normal(1., 1.), obs=data)

    expected_logprob = poutine.trace(hand_model).get_trace().log_prob_sum()
    actual_logprob = poutine.trace(collapse(model, -3)).get_trace().log_prob_sum() / num_particles

    assert_equal(expected_logprob, actual_logprob, prec=1e-3)

    p1 = pyro.param("p1")
    expected_grad = grad(expected_logprob, [p1,])[0]
    actual_grad = grad(actual_logprob, [p1,])[0]

    assert_equal(expected_grad, actual_grad, prec=1e-3)


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
            lambda msg: {"enumerate": "parallel", "collapse": True} if msg["name"] == "e" else {})

    collapsed_guide = collapse(guide, first_available_dim=-1)

    uncollapsed_tr = poutine.trace(guide).get_trace()
    collapsed_tr = poutine.trace(collapsed_guide).get_trace()

    assert set(collapsed_tr.nodes.keys()) == set(uncollapsed_tr.nodes.keys()) - {"e"}


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
        lambda msg: {"enumerate": "parallel", "collapse": True} if msg["name"] == "e" else {})

    collapsed_guide = collapse(guide, first_available_dim=-1)

    elbo = pyro.infer.TraceEnum_ELBO(
        max_plate_nesting=0, strict_enumeration_warning=False)
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
                infer={"collapse": True, "enumerate": "parallel"})
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

    elbo = pyro.infer.TraceEnum_ELBO(
        max_plate_nesting=1,  # XXX what should this be?
        strict_enumeration_warning=False)

    expected = elbo.differentiable_loss(model, guide, True)
    actual = elbo.differentiable_loss(model, collapsed_guide, False)

    assert_equal(expected, actual)


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
                infer={"collapse": True, "enumerate": "parallel"})
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

    elbo = pyro.infer.TraceEnum_ELBO(
        max_plate_nesting=2,  # XXX what should this be?
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
