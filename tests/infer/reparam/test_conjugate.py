# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import Trace_ELBO
from pyro.infer.reparam import ConjugateReparam
from tests.common import assert_close


def test_beta_binomial_sample():
    total = 10
    counts = dist.Binomial(total, 0.3).sample()
    concentration1 = torch.tensor(0.5)
    concentration0 = torch.tensor(1.5)

    prior = dist.Beta(concentration1, concentration0)
    likelihood = dist.Beta(1 + counts, 1 + total - counts)
    posterior = dist.Beta(concentration1 + counts, concentration0 + total - counts)

    def model(counts):
        prob = pyro.sample("prob", prior)
        pyro.sample("counts", dist.Binomial(total, prob), obs=counts)

    reparam_model = poutine.reparam(model, {"prob": ConjugateReparam(likelihood)})

    with poutine.trace() as tr, pyro.plate("particles", 100000):
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

    elbo = Trace_ELBO(num_particles=100000, vectorize_particles=True, max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model, guide)
    actual_loss = elbo.differentiable_loss(reparam_model, reparam_guide)
    assert_close(actual_loss, expected_loss, atol=0.01)

    params = [concentration1, concentration0]
    expected_grads = torch.autograd.grad(expected_loss, params, retain_graph=True)
    actual_grads = torch.autograd.grad(actual_loss, params, retain_graph=True)
    for a, e in zip(actual_grads, expected_grads):
        assert_close(a, e, rtol=0.01)
