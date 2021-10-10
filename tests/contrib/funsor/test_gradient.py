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
    from pyroapi import handlers, infer, pyro, pyro_backend
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
            num_particles=100000,
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

    assert_equal(actual_grads, expected_grads, prec=0.03)


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
            model_tr.nodes["a"]["log_prob"].data - guide_tr.nodes["a"]["log_prob"].data
        )
        loss_bc = (
            model_tr.nodes["b"]["log_prob"].data
            + model_tr.nodes["c"]["log_prob"].data
            - guide_tr.nodes["b"]["log_prob"].data
        )
        expected_grads = {
            "prob": -(dlogqa_dprob * (loss_a + loss_bc.sum()) - dlogqa_dprob),
            "rate": -(dlogqb_drate * (loss_bc) - dlogqb_drate),
        }
        actual_grads["rate"] = actual_grads["rate"][a.long()]

        for name in sorted(params):
            logger.info("expected {} = {}".format(name, expected_grads[name]))
            logger.info("actual   {} = {}".format(name, actual_grads[name]))

        assert_equal(actual_grads, expected_grads, prec=1e-4)


@pytest.mark.parametrize(
    "Elbo,backend",
    [
        ("TraceEnum_ELBO", "pyro"),
        ("Trace_ELBO", "contrib.funsor"),
        ("TraceGraph_ELBO", "pyro"),
    ],
)
def test_particle_gradient_2(Elbo, backend):
    with pyro_backend(backend):
        pyro.clear_param_store()
        data = torch.tensor([0.0, 1.0])

        def model():
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

        def guide():
            prob_a = pyro.param("prob_a", lambda: torch.tensor(0.5))
            prob_b = pyro.param("prob_b", lambda: torch.tensor([0.4, 0.3]))
            prob_c = pyro.param(
                "prob_c", lambda: torch.tensor([[0.3, 0.8], [0.2, 0.5]])
            )
            prob_d = pyro.param(
                "prob_d", lambda: torch.tensor([[0.2, 0.9], [0.1, 0.4]])
            )
            a = pyro.sample("a", dist.Bernoulli(prob_a))
            with pyro.plate("data", len(data)) as idx:
                b = pyro.sample("b", dist.Bernoulli(prob_b[a.long()]))
                pyro.sample("c", dist.Bernoulli(Vindex(prob_c)[b.long(), idx]))
                pyro.sample("d", dist.Bernoulli(Vindex(prob_d)[b.long(), idx]))

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
        c = guide_tr.nodes["c"]["value"].data

        proba = pyro.param("prob_a").data
        probb = pyro.param("prob_b").data
        probc = pyro.param("prob_c").data
        probd = pyro.param("prob_d").data

        logpa = model_tr.nodes["a"]["log_prob"].data
        logpba = model_tr.nodes["b"]["log_prob"].data
        logpcba = model_tr.nodes["c"]["log_prob"].data
        logpdba = model_tr.nodes["d"]["log_prob"].data
        logpecba = model_tr.nodes["e"]["log_prob"].data

        logqa = guide_tr.nodes["a"]["log_prob"].data
        logqba = guide_tr.nodes["b"]["log_prob"].data
        logqcba = guide_tr.nodes["c"]["log_prob"].data
        logqdba = guide_tr.nodes["d"]["log_prob"].data

        idx = torch.arange(2)
        dlogqa_dproba = (a - proba) / (proba * (1 - proba))
        dlogqb_dprobb = (b - probb[a.long()]) / (
            probb[a.long()] * (1 - probb[a.long()])
        )
        dlogqc_dprobc = (c - Vindex(probc)[b.long(), idx]) / (
            Vindex(probc)[b.long(), idx] * (1 - Vindex(probc)[b.long(), idx])
        )
        dlogqd_dprobd = (c - probd[b.long(), idx]) / (
            Vindex(probd)[b.long(), idx] * (1 - Vindex(probd)[b.long(), idx])
        )

        if Elbo == "Trace_ELBO":
            # fine-grained Rao-Blackwellization based on provenance tracking
            expected_grads = {
                "prob_a": -dlogqa_dproba
                * (
                    logpa
                    + (logpba + logpcba + logpdba + logpecba).sum()
                    - logqa
                    - (logqba + logqcba + logqdba).sum()
                    - 1
                ),
                "prob_b": (
                    -dlogqb_dprobb
                    * (
                        (logpba + logpcba + logpdba + logpecba)
                        - (logqba + logqcba + logqdba)
                        - 1
                    )
                ).sum(),
                "prob_c": -dlogqc_dprobc * (logpcba + logpecba - logqcba - 1),
                "prob_d": -dlogqd_dprobd * (logpdba - logqdba - 1),
            }
        elif Elbo == "TraceEnum_ELBO":
            # only uses plate conditional independence for Rao-Blackwellization
            expected_grads = {
                "prob_a": -dlogqa_dproba
                * (
                    logpa
                    + (logpba + logpcba + logpdba + logpecba).sum()
                    - logqa
                    - (logqba + logqcba + logqdba).sum()
                    - 1
                ),
                "prob_b": (
                    -dlogqb_dprobb
                    * (
                        (logpba + logpcba + logpdba + logpecba)
                        - (logqba + logqcba + logqdba)
                        - 1
                    )
                ).sum(),
                "prob_c": -dlogqc_dprobc
                * (
                    (logpba + logpcba + logpdba + logpecba)
                    - (logqba + logqcba + logqdba)
                    - 1
                ),
                "prob_d": -dlogqd_dprobd
                * (
                    (logpba + logpcba + logpdba + logpecba)
                    - (logqba + logqcba + logqdba)
                    - 1
                ),
            }
        elif Elbo == "TraceGraph_ELBO":
            # Raw-Blackwellization uses conditional independence based on
            # 1) the sequential order of samples
            # 2) plate generators
            # additionally removes dlogq_dparam terms
            expected_grads = {
                "prob_a": -dlogqa_dproba
                * (
                    logpa
                    + (logpba + logpcba + logpdba + logpecba).sum()
                    - logqa
                    - (logqba + logqcba + logqdba).sum()
                ),
                "prob_b": (
                    -dlogqb_dprobb
                    * (
                        (logpba + logpcba + logpdba + logpecba)
                        - (logqba + logqcba + logqdba)
                    )
                ).sum(),
                "prob_c": -dlogqc_dprobc
                * ((logpcba + logpdba + logpecba) - (logqcba + logqdba)),
                "prob_d": -dlogqd_dprobd * (logpdba + logpecba - logqdba),
            }
        actual_grads["prob_b"] = actual_grads["prob_b"][a.long()]
        actual_grads["prob_c"] = Vindex(actual_grads["prob_c"])[b.long(), idx]
        actual_grads["prob_d"] = Vindex(actual_grads["prob_d"])[b.long(), idx]

        for name in sorted(params):
            logger.info("expected {} = {}".format(name, expected_grads[name]))
            logger.info("actual   {} = {}".format(name, actual_grads[name]))

        assert_equal(actual_grads, expected_grads, prec=1e-4)
