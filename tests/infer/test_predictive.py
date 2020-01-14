# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal
from pyro.infer import Predictive, SVI, Trace_ELBO
from tests.common import assert_close


def model(num_trials):
    with pyro.plate("data", num_trials.size(0)):
        phi_prior = dist.Uniform(num_trials.new_tensor(0.), num_trials.new_tensor(1.))
        success_prob = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", dist.Binomial(num_trials, success_prob))


def one_hot_model(pseudocounts, classes=None):
    probs_prior = dist.Dirichlet(pseudocounts)
    probs = pyro.sample("probs", probs_prior)
    with pyro.plate("classes", classes.size(0) if classes is not None else 1, dim=-1):
        return pyro.sample("obs", dist.OneHotCategorical(probs), obs=classes)


def beta_guide(num_trials):
    phi_c0 = pyro.param("phi_c0", num_trials.new_tensor(5.0).expand([num_trials.size(0)]))
    phi_c1 = pyro.param("phi_c1", num_trials.new_tensor(5.0).expand([num_trials.size(0)]))
    with pyro.plate("data", num_trials.size(0)):
        phi_posterior = dist.Beta(concentration0=phi_c0, concentration1=phi_c1)
        pyro.sample("phi", phi_posterior)


@pytest.mark.parametrize("parallel", [False, True])
def test_posterior_predictive_svi_manual_guide(parallel):
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    svi = SVI(conditioned_model, beta_guide, optim.Adam(dict(lr=1.0)), Trace_ELBO())
    for i in range(1000):
        svi.step(num_trials)
    posterior_predictive = Predictive(model, guide=beta_guide, num_samples=10000,
                                      parallel=parallel, return_sites=["_RETURN"])
    marginal_return_vals = posterior_predictive.get_samples(num_trials)["_RETURN"]
    assert_close(marginal_return_vals.mean(dim=0), torch.ones(5) * 700, rtol=0.05)


@pytest.mark.parametrize("parallel", [False, True])
def test_posterior_predictive_svi_auto_delta_guide(parallel):
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    guide = AutoDelta(conditioned_model)
    svi = SVI(conditioned_model, guide, optim.Adam(dict(lr=1.0)), Trace_ELBO())
    for i in range(1000):
        svi.step(num_trials)
    posterior_predictive = Predictive(model, guide=guide, num_samples=10000, parallel=parallel)
    marginal_return_vals = posterior_predictive.get_samples(num_trials)["obs"]
    assert_close(marginal_return_vals.mean(dim=0), torch.ones(5) * 700, rtol=0.05)


@pytest.mark.parametrize("return_trace", [False, True])
def test_posterior_predictive_svi_auto_diag_normal_guide(return_trace):
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    guide = AutoDiagonalNormal(conditioned_model)
    svi = SVI(conditioned_model, guide, optim.Adam(dict(lr=0.1)), Trace_ELBO())
    for i in range(1000):
        svi.step(num_trials)
    posterior_predictive = Predictive(model, guide=guide, num_samples=10000, parallel=True)
    if return_trace:
        marginal_return_vals = posterior_predictive.get_vectorized_trace(num_trials).nodes["obs"]["value"]
    else:
        marginal_return_vals = posterior_predictive.get_samples(num_trials)["obs"]
    assert_close(marginal_return_vals.mean(dim=0), torch.ones(5) * 700, rtol=0.05)


def test_posterior_predictive_svi_one_hot():
    pseudocounts = torch.ones(3) * 0.1
    true_probs = torch.tensor([0.15, 0.6, 0.25])
    classes = dist.OneHotCategorical(true_probs).sample((10000,))
    guide = AutoDelta(one_hot_model)
    svi = SVI(one_hot_model, guide, optim.Adam(dict(lr=0.1)), Trace_ELBO())
    for i in range(1000):
        svi.step(pseudocounts, classes=classes)
    posterior_samples = Predictive(guide, num_samples=10000).get_samples(pseudocounts)
    posterior_predictive = Predictive(one_hot_model, posterior_samples)
    marginal_return_vals = posterior_predictive.get_samples(pseudocounts)["obs"]
    assert_close(marginal_return_vals.mean(dim=0), true_probs.unsqueeze(0), rtol=0.1)


@pytest.mark.parametrize("parallel", [False, True])
def test_shapes(parallel):
    num_samples = 10

    def model():
        x = pyro.sample("x", dist.Normal(0, 1).expand([2]).to_event(1))
        with pyro.plate("plate", 5):
            loc, log_scale = x.unbind(-1)
            y = pyro.sample("y", dist.Normal(loc, log_scale.exp()))
        return dict(x=x, y=y)

    guide = AutoDiagonalNormal(model)

    # Compute by hand.
    vectorize = pyro.plate("_vectorize", num_samples, dim=-2)
    trace = poutine.trace(vectorize(guide)).get_trace()
    expected = poutine.replay(vectorize(model), trace)()

    # Use Predictive.
    predictive = Predictive(model, guide=guide, return_sites=["x", "y"],
                            num_samples=num_samples, parallel=parallel)
    actual = predictive.get_samples()
    assert set(actual) == set(expected)
    assert actual["x"].shape == expected["x"].shape
    assert actual["y"].shape == expected["y"].shape


@pytest.mark.parametrize("with_plate", [True, False])
@pytest.mark.parametrize("event_shape", [(), (2,)])
def test_deterministic(with_plate, event_shape):
    def model(y=None):
        with pyro.util.optional(pyro.plate("plate", 3), with_plate):
            x = pyro.sample("x", dist.Normal(0, 1).expand(event_shape).to_event())
            x2 = pyro.deterministic("x2", x ** 2, event_dim=len(event_shape))

        pyro.deterministic("x3", x2)
        return pyro.sample("obs", dist.Normal(x2, 0.1).to_event(), obs=y)

    y = torch.tensor(4.)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(dict(lr=0.1)), Trace_ELBO())
    for i in range(100):
        svi.step(y)

    actual = Predictive(model, guide=guide, return_sites=["x2", "x3"], num_samples=1000)()
    x2_batch_shape = (3,) if with_plate else ()
    assert actual["x2"].shape == (1000,) + x2_batch_shape + event_shape
    # x3 shape is prepended 1 to match Pyro shape semantics
    x3_batch_shape = (1, 3) if with_plate else ()
    assert actual["x3"].shape == (1000,) + x3_batch_shape + event_shape
    assert_close(actual["x2"].mean(), y, rtol=0.1)
    assert_close(actual["x3"].mean(), y, rtol=0.1)
