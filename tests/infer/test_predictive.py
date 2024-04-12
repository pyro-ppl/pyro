# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer import SVI, MHResampler, Predictive, Trace_ELBO, WeighedPredictive
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal
from pyro.ops.stats import quantile, weighed_quantile
from tests.common import assert_close


def model(num_trials):
    with pyro.plate("data", num_trials.size(0)):
        phi_prior = dist.Uniform(num_trials.new_tensor(0.0), num_trials.new_tensor(1.0))
        success_prob = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", dist.Binomial(num_trials, success_prob))


def one_hot_model(pseudocounts, classes=None):
    probs_prior = dist.Dirichlet(pseudocounts)
    probs = pyro.sample("probs", probs_prior)
    with pyro.plate("classes", classes.size(0) if classes is not None else 1, dim=-1):
        return pyro.sample("obs", dist.OneHotCategorical(probs), obs=classes)


def beta_guide(num_trials):
    phi_c0 = pyro.param(
        "phi_c0", num_trials.new_tensor(5.0).expand([num_trials.size(0)])
    )
    phi_c1 = pyro.param(
        "phi_c1", num_trials.new_tensor(5.0).expand([num_trials.size(0)])
    )
    with pyro.plate("data", num_trials.size(0)):
        phi_posterior = dist.Beta(concentration0=phi_c0, concentration1=phi_c1)
        pyro.sample("phi", phi_posterior)


@pytest.mark.parametrize(
    "predictive, num_svi_steps, test_unweighed_convergence",
    [
        (Predictive, 5000, None),
        (WeighedPredictive, 5000, True),
        (WeighedPredictive, 1000, False),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_posterior_predictive_svi_manual_guide(
    parallel, predictive, num_svi_steps, test_unweighed_convergence
):
    true_probs = torch.ones(5) * 0.7
    num_trials = (
        torch.ones(5) * 400
    )  # Reduced to 400 from 1000 in order for guide optimization to converge
    num_samples = 10000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    elbo = Trace_ELBO(num_particles=100, vectorize_particles=True)
    svi = SVI(conditioned_model, beta_guide, optim.Adam(dict(lr=3.0)), elbo)
    for i in range(num_svi_steps):
        svi.step(num_trials)
    posterior_predictive = predictive(
        model,
        guide=beta_guide,
        num_samples=num_samples,
        parallel=parallel,
        return_sites=["_RETURN"],
    )
    if predictive is Predictive:
        marginal_return_vals = posterior_predictive(num_trials)["_RETURN"]
    else:
        weighed_samples = posterior_predictive(
            num_trials, model_guide=conditioned_model
        )
        marginal_return_vals = weighed_samples.samples["_RETURN"]
        assert marginal_return_vals.shape[:1] == weighed_samples.log_weights.shape
        # Resample weighed samples
        resampler = MHResampler(posterior_predictive)
        num_mh_steps = 10
        for mh_step_count in range(num_mh_steps):
            resampled_weighed_samples = resampler(
                num_trials, model_guide=conditioned_model
            )
        resampled_marginal_return_vals = resampled_weighed_samples.samples["_RETURN"]
        # Calculate CDF quantiles
        quantile_test_point = 0.95
        quantile_test_point_value = quantile(
            marginal_return_vals, [quantile_test_point]
        )[0]
        weighed_quantile_test_point_value = weighed_quantile(
            marginal_return_vals, [quantile_test_point], weighed_samples.log_weights
        )[0]
        resampled_quantile_test_point_value = quantile(
            resampled_marginal_return_vals, [quantile_test_point]
        )[0]
        logging.info(
            "Unweighed quantile at test point is: " + str(quantile_test_point_value)
        )
        logging.info(
            "Weighed quantile at test point is:   "
            + str(weighed_quantile_test_point_value)
        )
        logging.info(
            "Resampled quantile at test point is: "
            + str(resampled_quantile_test_point_value)
        )
        # Weighed and resampled quantiles should match
        assert_close(
            weighed_quantile_test_point_value,
            resampled_quantile_test_point_value,
            rtol=0.01,
        )
        if test_unweighed_convergence:
            # Weights should be uniform as the guide has the same distribution as the model
            assert weighed_samples.log_weights.std() < 0.6
            # Effective sample size should be close to actual number of samples taken from the guide
            assert weighed_samples.get_ESS() > 0.8 * num_samples
            # Weighed and unweighed quantiles should match if guide converged to true model
            assert_close(
                quantile_test_point_value,
                resampled_quantile_test_point_value,
                rtol=0.01,
            )
    assert_close(marginal_return_vals.mean(dim=0), torch.ones(5) * 280, rtol=0.1)


@pytest.mark.parametrize("predictive", [Predictive, WeighedPredictive])
@pytest.mark.parametrize("parallel", [False, True])
def test_posterior_predictive_svi_auto_delta_guide(parallel, predictive):
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    guide = AutoDelta(conditioned_model)
    svi = SVI(conditioned_model, guide, optim.Adam(dict(lr=1.0)), Trace_ELBO())
    for i in range(1000):
        svi.step(num_trials)
    posterior_predictive = predictive(
        model, guide=guide, num_samples=10000, parallel=parallel
    )
    if predictive is Predictive:
        marginal_return_vals = posterior_predictive.get_samples(num_trials)["obs"]
    else:
        weighed_samples = posterior_predictive.get_samples(
            num_trials, model_guide=conditioned_model
        )
        marginal_return_vals = weighed_samples.samples["obs"]
        assert marginal_return_vals.shape[:1] == weighed_samples.log_weights.shape
    assert_close(marginal_return_vals.mean(dim=0), torch.ones(5) * 700, rtol=0.05)


@pytest.mark.parametrize("predictive", [Predictive, WeighedPredictive])
@pytest.mark.parametrize("return_trace", [False, True])
def test_posterior_predictive_svi_auto_diag_normal_guide(return_trace, predictive):
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    guide = AutoDiagonalNormal(conditioned_model)
    svi = SVI(conditioned_model, guide, optim.Adam(dict(lr=0.1)), Trace_ELBO())
    for i in range(1000):
        svi.step(num_trials)
    posterior_predictive = predictive(
        model, guide=guide, num_samples=10000, parallel=True
    )
    if return_trace:
        marginal_return_vals = posterior_predictive.get_vectorized_trace(
            num_trials
        ).nodes["obs"]["value"]
    else:
        if predictive is Predictive:
            marginal_return_vals = posterior_predictive.get_samples(num_trials)["obs"]
        else:
            weighed_samples = posterior_predictive.get_samples(
                num_trials, model_guide=conditioned_model
            )
            marginal_return_vals = weighed_samples.samples["obs"]
            assert marginal_return_vals.shape[:1] == weighed_samples.log_weights.shape
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


@pytest.mark.parametrize("predictive", [Predictive, WeighedPredictive])
@pytest.mark.parametrize("parallel", [False, True])
def test_shapes(parallel, predictive):
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
    actual = predictive(
        model,
        guide=guide,
        return_sites=["x", "y"],
        num_samples=num_samples,
        parallel=parallel,
    )()
    if predictive is WeighedPredictive:
        assert actual.samples["x"].shape[:1] == actual.log_weights.shape
        assert actual.samples["y"].shape[:1] == actual.log_weights.shape
        actual = actual.samples
    assert set(actual) == set(expected)
    assert actual["x"].shape == expected["x"].shape
    assert actual["y"].shape == expected["y"].shape


@pytest.mark.parametrize("predictive", [Predictive, WeighedPredictive])
@pytest.mark.parametrize("with_plate", [True, False])
@pytest.mark.parametrize("event_shape", [(), (2,)])
def test_deterministic(with_plate, event_shape, predictive):
    def model(y=None):
        with pyro.util.optional(pyro.plate("plate", 3), with_plate):
            x = pyro.sample("x", dist.Normal(0, 1).expand(event_shape).to_event())
            x2 = pyro.deterministic("x2", x**2, event_dim=len(event_shape))

        pyro.deterministic("x3", x2)
        return pyro.sample("obs", dist.Normal(x2, 0.1).to_event(), obs=y)

    y = torch.tensor(4.0)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optim.Adam(dict(lr=0.1)), Trace_ELBO())
    for i in range(100):
        svi.step(y)

    actual = predictive(
        model, guide=guide, return_sites=["x2", "x3"], num_samples=1000
    )()
    if predictive is WeighedPredictive:
        assert actual.samples["x2"].shape[:1] == actual.log_weights.shape
        assert actual.samples["x3"].shape[:1] == actual.log_weights.shape
        actual = actual.samples
    x2_batch_shape = (3,) if with_plate else ()
    assert actual["x2"].shape == (1000,) + x2_batch_shape + event_shape
    # x3 shape is prepended 1 to match Pyro shape semantics
    x3_batch_shape = (1, 3) if with_plate else ()
    assert actual["x3"].shape == (1000,) + x3_batch_shape + event_shape
    assert_close(actual["x2"].mean(), y, rtol=0.1)
    assert_close(actual["x3"].mean(), y, rtol=0.1)


def test_get_mask_optimization():
    def model():
        x = pyro.sample("x", dist.Normal(0, 1))
        pyro.sample("y", dist.Normal(x, 1), obs=torch.tensor(0.0))
        called.add("model-always")
        if poutine.get_mask() is not False:
            called.add("model-sometimes")
            pyro.factor("f", x + 1)

    def guide():
        x = pyro.sample("x", dist.Normal(0, 1))
        called.add("guide-always")
        if poutine.get_mask() is not False:
            called.add("guide-sometimes")
            pyro.factor("g", 2 - x)

    called = set()
    trace = poutine.trace(guide).get_trace()
    poutine.replay(model, trace)()
    assert "model-always" in called
    assert "guide-always" in called
    assert "model-sometimes" in called
    assert "guide-sometimes" in called

    called = set()
    with poutine.mask(mask=False):
        trace = poutine.trace(guide).get_trace()
        poutine.replay(model, trace)()
    assert "model-always" in called
    assert "guide-always" in called
    assert "model-sometimes" not in called
    assert "guide-sometimes" not in called

    called = set()
    Predictive(model, guide=guide, num_samples=2, parallel=True)()
    assert "model-always" in called
    assert "guide-always" in called
    assert "model-sometimes" not in called
    assert "guide-sometimes" not in called
