# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pyroapi
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
    from pyroapi import handlers, infer, pyro
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
    with pyroapi.pyro_backend("pyro"):
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
    with pyroapi.pyro_backend("contrib.funsor"):
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

    assert_equal(actual_grads, expected_grads, prec=0.02)


@pyroapi.pyro_backend("contrib.funsor")
def test_particle_gradient_0():
    # model
    # +---------+
    # | z --> x |
    # +---------+
    #
    # guide
    # +---+
    # | z |
    # +---+
    data = torch.tensor([-0.5, 2.0])

    def model():
        with pyro.plate("data", len(data)):
            z = pyro.sample("z", dist.Poisson(3))
            pyro.sample("x", dist.Normal(z, 1), obs=data)

    def guide():
        # set this to ensure rng agrees across runs
        # this should be ok since we are comparing a single particle gradients
        pyro.set_rng_seed(0)
        with pyro.plate("data", len(data)):
            rate = pyro.param("rate", lambda: torch.tensor([3.5, 1.5]))
            pyro.sample("z", dist.Poisson(rate))

    elbo = infer.Trace_ELBO(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
        num_particles=1,
        strict_enumeration_warning=False,
    )

    # Trace_ELBO gradients
    pyro.clear_param_store()
    elbo.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}

    # Hand derived gradients
    # elbo = Expectation(
    #   sum(dice_factor_zi * (log_pzi + log_pxi - log_qzi))
    # )
    pyro.clear_param_store()
    guide_tr = handlers.trace(guide).get_trace()
    model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
    guide_tr.compute_log_prob()
    model_tr.compute_log_prob()
    # log factors
    logpx = model_tr.nodes["x"]["log_prob"]
    logpz = model_tr.nodes["z"]["log_prob"]
    logqz = guide_tr.nodes["z"]["log_prob"]
    # dice factor
    df_z = (logqz - logqz.detach()).exp()
    # dice elbo
    dice_elbo = (df_z * (logpz + logpx - logqz)).sum()
    # backward run
    loss = -dice_elbo
    loss.backward()
    params = dict(pyro.get_param_store().named_parameters())
    expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=1e-4)


@pyroapi.pyro_backend("contrib.funsor")
def test_particle_gradient_1():
    # model
    #    +-----------+
    # a -|-> b --> c |
    #    +-----------+
    #
    # guide
    #    +-----+
    # a -|-> b |
    #    +-----+
    data = torch.tensor([-0.5, 2.0])

    def model():
        a = pyro.sample("a", dist.Bernoulli(0.3))
        with pyro.plate("data", len(data)):
            rate = torch.tensor([2.0, 3.0])
            b = pyro.sample("b", dist.Poisson(rate[a.long()]))
            pyro.sample("c", dist.Normal(b, 1), obs=data)

    def guide():
        # set this to ensure rng agrees across runs
        # this should be ok since we are comparing a single particle gradients
        pyro.set_rng_seed(0)
        prob = pyro.param(
            "prob",
            lambda: torch.tensor(0.5),
        )
        a = pyro.sample("a", dist.Bernoulli(prob))
        with pyro.plate("data", len(data)):
            rate = pyro.param("rate", lambda: torch.tensor([[3.5, 1.5], [0.5, 2.5]]))
            pyro.sample("b", dist.Poisson(rate[a.long()]))

    elbo = infer.Trace_ELBO(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
        num_particles=1,
        strict_enumeration_warning=False,
    )

    # Trace_ELBO gradients
    pyro.clear_param_store()
    elbo.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}

    # Hand derived gradients
    # elbo = Expectation(
    #   q(a) * log_pa
    #   + q(a) * sum(q(b_i|a) * log_pb_i)
    #   + q(a) * sum(q(b_i|a) * log_pc_i)
    #   - q(a) * log_qa
    #   - q(a) * sum(q(b_i|a) * log_qb_i)
    # )
    pyro.clear_param_store()
    guide_tr = handlers.trace(guide).get_trace()
    model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
    guide_tr.compute_log_prob()
    model_tr.compute_log_prob()
    # log factors
    logpa = model_tr.nodes["a"]["log_prob"]
    logpb = model_tr.nodes["b"]["log_prob"]
    logpc = model_tr.nodes["c"]["log_prob"]
    logqa = guide_tr.nodes["a"]["log_prob"]
    logqb = guide_tr.nodes["b"]["log_prob"]
    # dice factors
    df_a = (logqa - logqa.detach()).exp()
    df_b = (logqb - logqb.detach()).exp()
    # dice elbo
    dice_elbo = (
        df_a * logpa
        + df_a * (df_b * logpb).sum()
        + df_a * (df_b * logpc).sum()
        - df_a * logqa
        - df_a * (df_b * logqb).sum()
    )
    # backward run
    loss = -dice_elbo
    loss.backward()
    params = dict(pyro.get_param_store().named_parameters())
    expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=1e-4)


@pyroapi.pyro_backend("contrib.funsor")
def test_particle_gradient_2():
    # model
    #    +-----------------+
    # a -|-> b --> c --> e |
    #    |    \--> d       |
    #    +-----------------+
    #
    # guide
    #    +-----------+
    # a -|-> b --> c |
    #    |    \--> d |
    #    +-----------+
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
        # set this to ensure rng agrees across runs
        # this should be ok since we are comparing a single particle gradients
        pyro.set_rng_seed(0)
        prob_a = pyro.param("prob_a", lambda: torch.tensor(0.5))
        prob_b = pyro.param("prob_b", lambda: torch.tensor([0.4, 0.3]))
        prob_c = pyro.param("prob_c", lambda: torch.tensor([[0.3, 0.8], [0.2, 0.5]]))
        prob_d = pyro.param("prob_d", lambda: torch.tensor([[0.2, 0.9], [0.1, 0.4]]))
        a = pyro.sample("a", dist.Bernoulli(prob_a))
        with pyro.plate("data", len(data)) as idx:
            b = pyro.sample("b", dist.Bernoulli(prob_b[a.long()]))
            pyro.sample("c", dist.Bernoulli(Vindex(prob_c)[b.long(), idx]))
            pyro.sample("d", dist.Bernoulli(Vindex(prob_d)[b.long(), idx]))

    elbo = infer.Trace_ELBO(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
        num_particles=1,
        strict_enumeration_warning=False,
    )

    # Trace_ELBO gradients
    pyro.clear_param_store()
    elbo.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}

    # Hand derived gradients
    # elbo = Expectation(
    #   q(a) * log_pa
    #   + q(a) * sum(q(b_i|a) * log_pb_i)
    #   + q(a) * sum(q(b_i|a) * q(c_i|b_i) * log_pc_i)
    #   + q(a) * sum(q(b_i|a) * q(c_i|b_i) * log_pe_i)
    #   + q(a) * sum(q(b_i|a) * q(d_i|b_i) * log_pd_i)
    #   - q(a) * log_qa
    #   - q(a) * sum(q(b_i|a) * log_qb_i)
    #   - q(a) * sum(q(b_i|a) * q(c_i|b_i) * log_pq_i)
    #   - q(a) * sum(q(b_i|a) * q(d_i|b_i) * log_pd_i)
    # )
    pyro.clear_param_store()
    guide_tr = handlers.trace(guide).get_trace()
    model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
    guide_tr.compute_log_prob()
    model_tr.compute_log_prob()
    # log factors
    logpa = model_tr.nodes["a"]["log_prob"]
    logpb = model_tr.nodes["b"]["log_prob"]
    logpc = model_tr.nodes["c"]["log_prob"]
    logpd = model_tr.nodes["d"]["log_prob"]
    logpe = model_tr.nodes["e"]["log_prob"]

    logqa = guide_tr.nodes["a"]["log_prob"]
    logqb = guide_tr.nodes["b"]["log_prob"]
    logqc = guide_tr.nodes["c"]["log_prob"]
    logqd = guide_tr.nodes["d"]["log_prob"]
    # dice factors
    df_a = (logqa - logqa.detach()).exp()
    df_b = (logqb - logqb.detach()).exp()
    df_c = (logqc - logqc.detach()).exp()
    df_d = (logqd - logqd.detach()).exp()
    # dice elbo
    dice_elbo = (
        df_a * logpa
        + df_a * (df_b * logpb).sum()
        + df_a * (df_b * df_c * logpc).sum()
        + df_a * (df_b * df_c * logpe).sum()
        + df_a * (df_b * df_d * logpd).sum()
        - df_a * logqa
        - df_a * (df_b * logqb).sum()
        - df_a * (df_b * df_c * logqc).sum()
        - df_a * (df_b * df_d * logqd).sum()
    )
    # backward run
    loss = -dice_elbo
    loss.backward()
    params = dict(pyro.get_param_store().named_parameters())
    expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=1e-4)
