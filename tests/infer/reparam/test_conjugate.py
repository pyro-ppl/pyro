# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.reparam import ConjugateReparam
from tests.common import assert_close


def test_beta_binomial():
    total = 10
    counts = dist.Binomial(total, 0.3).sample()
    concentration1 = torch.tensor(0.5)
    concentration0 = torch.tensor(1.5)
    prior = dist.Beta(concentration1, concentration0)

    def model(counts):
        prob = pyro.sample("prob", prior)
        pyro.sample("counts", dist.Binomial(total, prob), obs=counts)

    reparam_model = poutine.reparam(model, {
        "prob": ConjugateReparam(dist.Beta(1 + counts, 1 + total - counts))
    })

    with poutine.trace() as tr, pyro.plate("particles", 100000):
        reparam_model(counts)
    samples = tr.trace.nodes["prob"]["value"]

    posterior = dist.Beta(concentration1 + counts, concentration0 + total - counts)
    assert_close(samples.mean(), posterior.mean, atol=0.01)
    assert_close(samples.std(), posterior.variance.sqrt(), atol=0.01)
