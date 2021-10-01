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
        z = pyro.sample("z", dist.Categorical(torch.tensor([0.3, 0.7])))
        pyro.sample("x", dist.Normal(z.to(data.dtype), 1), obs=data)


def guide_0(data):
    with pyro.plate("data", len(data)):
        probs = pyro.param("probs", lambda: torch.tensor([[0.4, 0.6], [0.5, 0.5]]))
        pyro.sample("z", dist.Categorical(probs))


def model_1(data):
    a = pyro.sample("a", dist.Categorical(torch.tensor([0.3, 0.7])))
    with pyro.plate("data", len(data)):
        probs_b = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        b = pyro.sample("b", dist.Categorical(probs_b[a.long()]))
        pyro.sample("c", dist.Normal(b.to(data.dtype), 1), obs=data)


def guide_1(data):
    probs_a = pyro.param(
        "probs_a",
        lambda: torch.tensor([0.5, 0.5]),
    )
    a = pyro.sample("a", dist.Categorical(probs_a))
    with pyro.plate("data", len(data)) as idx:
        probs_b = pyro.param(
            "probs_b",
            lambda: torch.tensor(
                [[[0.5, 0.5], [0.6, 0.4]], [[0.4, 0.6], [0.35, 0.65]]]
            ),
        )
        pyro.sample("b", dist.Categorical(Vindex(probs_b)[a.long(), idx]))


def model_2(data):
    prob_b = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    prob_c = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
    prob_d = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
    prob_e = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
    a = pyro.sample("a", dist.Categorical(torch.tensor([0.3, 0.7])))
    with pyro.plate("data", len(data)):
        b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
        c = pyro.sample("c", dist.Categorical(prob_c[b.long()]))
        pyro.sample("d", dist.Categorical(prob_d[b.long()]))
        pyro.sample("e", dist.Categorical(prob_e[c.long()]), obs=data)


def guide_2(data):
    prob_a = pyro.param("prob_a", lambda: torch.tensor([0.5, 0.5]))
    prob_b = pyro.param("prob_b", lambda: torch.tensor([[0.4, 0.6], [0.3, 0.7]]))
    prob_c = pyro.param(
        "prob_c",
        lambda: torch.tensor([[[0.3, 0.7], [0.8, 0.2]], [[0.2, 0.8], [0.5, 0.5]]]),
    )
    prob_d = pyro.param(
        "prob_d",
        lambda: torch.tensor([[[0.2, 0.8], [0.9, 0.1]], [[0.1, 0.9], [0.4, 0.6]]]),
    )
    a = pyro.sample("a", dist.Categorical(prob_a))
    with pyro.plate("data", len(data)) as idx:
        b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
        pyro.sample("c", dist.Categorical(Vindex(prob_c)[b.long(), idx]))
        pyro.sample("d", dist.Categorical(Vindex(prob_d)[b.long(), idx]))


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


@pytest.mark.parametrize(
    "model,guide,data",
    [
        (model_0, guide_0, torch.tensor([-0.5, 2.0])),
        (model_1, guide_1, torch.tensor([-0.5, 2.0])),
        (model_2, guide_2, torch.tensor([0.0, 1.0])),
    ],
)
def test_guide_enum_gradient(model, guide, data):

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

    # Actual grads
    with pyro_backend("contrib.funsor"):
        pyro.clear_param_store()
        elbo = infer.Trace_ELBO(
            max_plate_nesting=1,  # set this to ensure rng agrees across runs
            strict_enumeration_warning=False,
        )
        elbo.loss_and_grads(model, infer.config_enumerate(guide), data)
        params = dict(pyro.get_param_store().named_parameters())
        actual_grads = {
            name: param.grad.detach().cpu() for name, param in params.items()
        }

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=1e-4)
