# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

from pyro.ops.indexing import Vindex
from tests.common import assert_equal

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import infer, pyro, pyro_backend
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)


def model_0(data):
    with pyro.plate("data", len(data)):
        z = pyro.sample("z", dist.Bernoulli(0.3))
        pyro.sample("x", dist.Normal(z, 1), obs=data)


def guide_0(data):
    with pyro.plate("data", len(data)):
        probs = pyro.param("probs", lambda: torch.tensor([0.4, 0.5]))
        pyro.sample("z", dist.Bernoulli(probs))


def model_1(data):
    a = pyro.sample("a", dist.Bernoulli(0.3))
    with pyro.plate("data", len(data)):
        probs_b = torch.tensor([0.1, 0.2])
        b = pyro.sample("b", dist.Bernoulli(probs_b[a.long()]))
        pyro.sample("c", dist.Normal(b, 1), obs=data)


def guide_1(data):
    probs_a = pyro.param(
        "probs_a",
        lambda: torch.tensor(0.5),
    )
    a = pyro.sample("a", dist.Bernoulli(probs_a))
    with pyro.plate("data", len(data)):
        probs_b = pyro.param("probs_b", lambda: torch.tensor([[0.5, 0.6], [0.4, 0.35]]))
        pyro.sample("b", dist.Bernoulli(Vindex(probs_b)[a.long(), torch.arange(2)]))


def model_2(data):
    prob_b = torch.tensor([0.3, 0.4])
    prob_c = torch.tensor([0.5, 0.6])
    prob_d = torch.tensor([0.2, 0.3])
    prob_e = torch.tensor([0.5, 0.1])
    a = pyro.sample("a", dist.Bernoulli(0.3))
    with pyro.plate("data", len(data)):
        b = pyro.sample("b", dist.Bernoulli(prob_b[a.long()]))
        c = pyro.sample("c", dist.Bernoulli(prob_c[b.long()]))
        pyro.sample("d", dist.Bernoulli(prob_d[b.long()]))
        pyro.sample("e", dist.Bernoulli(prob_e[c.long()]), obs=data)


def guide_2(data):
    prob_a = pyro.param("prob_a", lambda: torch.tensor(0.5))
    prob_b = pyro.param("prob_b", lambda: torch.tensor([0.4, 0.3]))
    prob_c = pyro.param("prob_c", lambda: torch.tensor([[0.3, 0.8], [0.2, 0.5]]))
    prob_d = pyro.param("prob_d", lambda: torch.tensor([[0.2, 0.9], [0.1, 0.4]]))
    a = pyro.sample("a", dist.Bernoulli(prob_a))
    with pyro.plate("data", len(data)) as idx:
        b = pyro.sample("b", dist.Bernoulli(prob_b[a.long()]))
        pyro.sample("c", dist.Bernoulli(Vindex(prob_c)[b.long(), idx]))
        pyro.sample("d", dist.Bernoulli(Vindex(prob_d)[b.long(), idx]))


@pytest.mark.parametrize(
    "model,guide,data",
    [
        (model_0, guide_0, torch.tensor([-0.5, 2.0])),
        (model_1, guide_1, torch.tensor([-0.5, 2.0])),
        (model_2, guide_2, torch.tensor([0.0, 1.0])),
    ],
)
def test_gradient(model, guide, data):

    # Expected grads based on exact integration
    with pyro_backend("pyro"):
        pyro.clear_param_store()
        elbo = infer.TraceEnum_ELBO(
            max_plate_nesting=1,  # set this to ensure rng agrees across runs
            strict_enumeration_warning=False,
        )
        elbo.loss_and_grads(model, infer.config_enumerate(guide), data)
        params = dict(pyro.get_param_store().named_parameters())
        expected_grads = {
            name: param.grad.detach().cpu() for name, param in params.items()
        }

    # Actual grads averaged over num_particles
    with pyro_backend("contrib.funsor"):
        pyro.clear_param_store()
        elbo = infer.Trace_ELBO(
            max_plate_nesting=1,  # set this to ensure rng agrees across runs
            num_particles=50000,
            vectorize_particles=True,
            strict_enumeration_warning=False,
        )
        elbo.loss_and_grads(model, guide, data)
        params = dict(pyro.get_param_store().named_parameters())
        actual_grads = {
            name: param.grad.detach().cpu() for name, param in params.items()
        }

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=0.06)
