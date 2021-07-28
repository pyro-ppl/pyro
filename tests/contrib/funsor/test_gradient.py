# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch
from torch.distributions import constraints

from tests.common import assert_equal

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import handlers, infer, pyro, pyro_backend
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)

# _PYRO_BACKEND = os.environ.get("TEST_ENUM_PYRO_BACKEND", "contrib.funsor")


@pytest.mark.parametrize(
    "Elbo,backend",
    [
        ("TraceEnum_ELBO", "pyro"),
        ("Trace_ELBO", "contrib.funsor"),
    ],
)
def test_particle_gradient_0(Elbo, backend):
    with pyro_backend(backend):
        pyro.clear_param_store()
        data = torch.tensor([-0.5, 2.0])

        def model():
            with pyro.plate("data", len(data)):
                z = pyro.sample("z", dist.Poisson(3))
                pyro.sample("x", dist.Normal(z, 1), obs=data)

        def guide():
            with pyro.plate("data", len(data)):
                rate = pyro.param("rate", lambda: torch.tensor([3.5, 1.5]))
                pyro.sample("z", dist.Poisson(rate))

        elbo = getattr(infer, Elbo)(
            max_plate_nesting=1,  # set this to ensure rng agrees across runs
            num_particles=1,
            strict_enumeration_warning=False,
        )

        # Elbo gradient estimator
        pyro.set_rng_seed(0)
        elbo.loss_and_grads(model, guide)
        params = dict(pyro.get_param_store().named_parameters())
        actual_grads = {
            name: param.grad.detach().cpu() for name, param in params.items()
        }

        # capture sample values and log_probs
        pyro.set_rng_seed(0)
        guide_tr = handlers.trace(guide).get_trace()
        model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
        guide_tr.compute_log_prob()
        model_tr.compute_log_prob()
        z = guide_tr.nodes["z"]["value"].data
        rate = pyro.param("rate").data

        loss_i = (
            model_tr.nodes["x"]["log_prob"].data
            + model_tr.nodes["z"]["log_prob"].data
            - guide_tr.nodes["z"]["log_prob"].data
        )
        dlogq_drate = z / rate - 1
        expected_grads = {
            "rate": -(dlogq_drate * loss_i - dlogq_drate),
        }

        for name in sorted(params):
            logger.info("expected {} = {}".format(name, expected_grads[name]))
            logger.info("actual   {} = {}".format(name, actual_grads[name]))

        assert_equal(actual_grads, expected_grads, prec=1e-4)


@pytest.mark.parametrize(
    "Elbo,backend",
    [
        ("TraceEnum_ELBO", "pyro"),
        ("Trace_ELBO", "contrib.funsor"),
    ],
)
def test_particle_gradient_1(Elbo, backend):
    with pyro_backend(backend):
        pyro.clear_param_store()
        data = torch.tensor([-0.5, 2.0])

        def model():
            a = pyro.sample("a", dist.Bernoulli(0.3))
            with pyro.plate("data", len(data)):
                rate = torch.tensor([2.0, 3.0])
                b = pyro.sample("b", dist.Poisson(rate[a.long()]))
                pyro.sample("c", dist.Normal(b, 1), obs=data)

        def guide():
            prob = pyro.param(
                "prob",
                lambda: torch.tensor(0.5),
                # constraint=constraints.unit_interval,
            )
            a = pyro.sample("a", dist.Bernoulli(prob))
            with pyro.plate("data", len(data)):
                rate = pyro.param(
                    "rate", lambda: torch.tensor([[3.5, 1.5], [0.5, 2.5]])
                )
                pyro.sample("b", dist.Poisson(rate[a.long()]))

        elbo = getattr(infer, Elbo)(
            max_plate_nesting=1,  # set this to ensure rng agrees across runs
            num_particles=1,
            strict_enumeration_warning=False,
        )

        # Elbo gradient estimator
        pyro.set_rng_seed(0)
        elbo.loss_and_grads(model, guide)
        params = dict(pyro.get_param_store().named_parameters())
        actual_grads = {
            name: param.grad.detach().cpu() for name, param in params.items()
        }

        # capture sample values and log_probs
        pyro.set_rng_seed(0)
        guide_tr = handlers.trace(guide).get_trace()
        model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
        guide_tr.compute_log_prob()
        model_tr.compute_log_prob()
        a = guide_tr.nodes["a"]["value"].data
        b = guide_tr.nodes["b"]["value"].data
        prob = pyro.param("prob").data
        rate = pyro.param("rate").data

        dlogqa_dprob = (a - prob) / (prob * (1 - prob))
        dlogqb_drate = b / rate[a.long()] - 1

        loss_a = (
            model_tr.nodes["a"]["log_prob"].data
            - guide_tr.nodes["a"]["log_prob"].data
        )
        loss_bc = (
            model_tr.nodes["b"]["log_prob"].data
            + model_tr.nodes["c"]["log_prob"].data
            - guide_tr.nodes["b"]["log_prob"].data
        )
        # dlogq_drate = z / rate - 1
        expected_grads = {
            "prob": -(dlogqa_dprob * (loss_a + loss_bc.sum()) - dlogqa_dprob),
            "rate": -(dlogqb_drate * (loss_bc) - dlogqb_drate),
        }
        actual_grads["rate"] = actual_grads["rate"][a.long()]

        for name in sorted(params):
            logger.info("expected {} = {}".format(name, expected_grads[name]))
            logger.info("actual   {} = {}".format(name, actual_grads[name]))

        assert_equal(actual_grads, expected_grads, prec=1e-4)
