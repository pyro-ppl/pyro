# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.reparam import ConjugateReparam, LinearHMMReparam, StableReparam
from tests.common import assert_close
from tests.ops.gaussian import random_mvn


def test_beta_binomial_static_sample():
    total = 10
    counts = dist.Binomial(total, 0.3).sample()
    concentration1 = torch.tensor(0.5)
    concentration0 = torch.tensor(1.5)

    prior = dist.Beta(concentration1, concentration0)
    likelihood = dist.Beta(1 + counts, 1 + total - counts)
    posterior = dist.Beta(concentration1 + counts, concentration0 + total - counts)

    def model():
        prob = pyro.sample("prob", prior)
        pyro.sample("counts", dist.Binomial(total, prob), obs=counts)

    reparam_model = poutine.reparam(model, {"prob": ConjugateReparam(likelihood)})

    with poutine.trace() as tr, pyro.plate("particles", 10000):
        reparam_model()
    samples = tr.trace.nodes["prob"]["value"]

    assert_close(samples.mean(), posterior.mean, atol=0.01)
    assert_close(samples.std(), posterior.variance.sqrt(), atol=0.01)


def test_beta_binomial_dependent_sample():
    total = 10
    counts = dist.Binomial(total, 0.3).sample()
    concentration1 = torch.tensor(0.5)
    concentration0 = torch.tensor(1.5)

    prior = dist.Beta(concentration1, concentration0)
    posterior = dist.Beta(concentration1 + counts, concentration0 + total - counts)

    def model(counts):
        prob = pyro.sample("prob", prior)
        pyro.sample("counts", dist.Binomial(total, prob), obs=counts)

    reparam_model = poutine.reparam(model, {
        "prob": ConjugateReparam(lambda counts: dist.Beta(1 + counts, 1 + total - counts)),
    })

    with poutine.trace() as tr, pyro.plate("particles", 10000):
        reparam_model(counts)
    samples = tr.trace.nodes["prob"]["value"]

    assert_close(samples.mean(), posterior.mean, atol=0.01)
    assert_close(samples.std(), posterior.variance.sqrt(), atol=0.01)


def test_beta_binomial_elbo():
    total = 10
    counts = dist.Binomial(total, 0.3).sample()
    concentration1 = torch.tensor(0.5, requires_grad=True)
    concentration0 = torch.tensor(1.5, requires_grad=True)

    prior = dist.Beta(concentration1, concentration0)
    likelihood = dist.Beta(1 + counts, 1 + total - counts)
    posterior = dist.Beta(concentration1 + counts, concentration0 + total - counts)

    def model():
        prob = pyro.sample("prob", prior)
        pyro.sample("counts", dist.Binomial(total, prob), obs=counts)

    def guide():
        pyro.sample("prob", posterior)

    reparam_model = poutine.reparam(model, {"prob": ConjugateReparam(likelihood)})

    def reparam_guide():
        pass

    elbo = Trace_ELBO(num_particles=10000, vectorize_particles=True, max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model, guide)
    actual_loss = elbo.differentiable_loss(reparam_model, reparam_guide)
    assert_close(actual_loss, expected_loss, atol=0.01)

    params = [concentration1, concentration0]
    expected_grads = torch.autograd.grad(expected_loss, params, retain_graph=True)
    actual_grads = torch.autograd.grad(actual_loss, params, retain_graph=True)
    for a, e in zip(actual_grads, expected_grads):
        assert_close(a, e, rtol=0.01)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("hidden_dim,obs_dim", [(1, 1), (3, 2)], ids=str)
@pytest.mark.parametrize("num_steps", range(1, 6))
def test_gaussian_hmm_elbo(batch_shape, num_steps, hidden_dim, obs_dim):
    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_mat = torch.randn(batch_shape + (num_steps, hidden_dim, hidden_dim), requires_grad=True)
    trans_dist = random_mvn(batch_shape + (num_steps,), hidden_dim)
    obs_mat = torch.randn(batch_shape + (num_steps, hidden_dim, obs_dim), requires_grad=True)
    obs_dist = random_mvn(batch_shape + (num_steps,), obs_dim)

    data = obs_dist.sample()
    assert data.shape == batch_shape + (num_steps, obs_dim)
    prior = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)
    likelihood = dist.Normal(data, 1).to_event(2)
    posterior, log_normalizer = prior.conjugate_update(likelihood)

    def model(data):
        with pyro.plate_stack("plates", batch_shape):
            z = pyro.sample("z", prior)
            pyro.sample("x", dist.Normal(z, 1).to_event(2), obs=data)

    def guide(data):
        with pyro.plate_stack("plates", batch_shape):
            pyro.sample("z", posterior)

    reparam_model = poutine.reparam(model, {"z": ConjugateReparam(likelihood)})

    def reparam_guide(data):
        pass

    elbo = Trace_ELBO(num_particles=1000, vectorize_particles=True)
    expected_loss = elbo.differentiable_loss(model, guide, data)
    actual_loss = elbo.differentiable_loss(reparam_model, reparam_guide, data)
    assert_close(actual_loss, expected_loss, atol=0.01)

    params = [trans_mat, obs_mat]
    expected_grads = torch.autograd.grad(expected_loss, params, retain_graph=True)
    actual_grads = torch.autograd.grad(actual_loss, params, retain_graph=True)
    for a, e in zip(actual_grads, expected_grads):
        assert_close(a, e, rtol=0.01)


def random_stable(shape):
    stability = dist.Uniform(1.4, 1.9).sample(shape)
    skew = dist.Uniform(-1, 1).sample(shape)
    scale = torch.rand(shape).exp()
    loc = torch.randn(shape)
    return dist.Stable(stability, skew, scale, loc)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("hidden_dim,obs_dim", [(1, 1), (2, 3)], ids=str)
@pytest.mark.parametrize("num_steps", range(1, 6))
def test_stable_hmm_smoke(batch_shape, num_steps, hidden_dim, obs_dim):
    init_dist = random_stable(batch_shape + (hidden_dim,)).to_event(1)
    trans_mat = torch.randn(batch_shape + (num_steps, hidden_dim, hidden_dim), requires_grad=True)
    trans_dist = random_stable(batch_shape + (num_steps, hidden_dim)).to_event(1)
    obs_mat = torch.randn(batch_shape + (num_steps, hidden_dim, obs_dim), requires_grad=True)
    obs_dist = random_stable(batch_shape + (num_steps, obs_dim)).to_event(1)
    data = obs_dist.sample()
    assert data.shape == batch_shape + (num_steps, obs_dim)

    def model(data):
        hmm = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=num_steps)
        with pyro.plate_stack("plates", batch_shape):
            z = pyro.sample("z", hmm)
            pyro.sample("x", dist.Normal(z, 1).to_event(2), obs=data)

    # Test that we can combine these two reparameterizers.
    reparam_model = poutine.reparam(model, {
        "z": LinearHMMReparam(StableReparam(), StableReparam(), StableReparam()),
    })
    reparam_model = poutine.reparam(reparam_model, {
        "z": ConjugateReparam(dist.Normal(data, 1).to_event(2)),
    })
    reparam_guide = AutoDiagonalNormal(reparam_model)  # Models auxiliary variables.

    # Smoke test only.
    elbo = Trace_ELBO(num_particles=5, vectorize_particles=True)
    loss = elbo.differentiable_loss(reparam_model, reparam_guide, data)
    params = [trans_mat, obs_mat]
    torch.autograd.grad(loss, params, retain_graph=True)


def test_beta_binomial_hmc():
    num_samples = 1000
    total = 10
    counts = dist.Binomial(total, 0.3).sample()
    concentration1 = torch.tensor(0.5)
    concentration0 = torch.tensor(1.5)

    prior = dist.Beta(concentration1, concentration0)
    likelihood = dist.Beta(1 + counts, 1 + total - counts)
    posterior = dist.Beta(concentration1 + counts, concentration0 + total - counts)

    def model():
        prob = pyro.sample("prob", prior)
        pyro.sample("counts", dist.Binomial(total, prob), obs=counts)

    reparam_model = poutine.reparam(model, {"prob": ConjugateReparam(likelihood)})

    kernel = HMC(reparam_model)
    samples = MCMC(kernel, num_samples, warmup_steps=0).run()
    pred = Predictive(reparam_model, samples, num_samples=num_samples)
    trace = pred.get_vectorized_trace()
    samples = trace.nodes["prob"]["value"]

    assert_close(samples.mean(), posterior.mean, atol=0.01)
    assert_close(samples.std(), posterior.variance.sqrt(), atol=0.01)
