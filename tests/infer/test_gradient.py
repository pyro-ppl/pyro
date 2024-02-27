# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest
import torch
import torch.optim
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.testing import fakes
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    JitTraceGraph_ELBO,
    JitTraceMeanField_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceGraph_ELBO,
    TraceMeanField_ELBO,
    config_enumerate,
)
from pyro.optim import Adam
from tests.common import assert_equal, xfail_if_not_implemented, xfail_param

logger = logging.getLogger(__name__)


def DiffTrace_ELBO(*args, **kwargs):
    return Trace_ELBO(*args, **kwargs).differentiable_loss


@pytest.mark.parametrize(
    "reparameterized,has_rsample",
    [(True, None), (True, False), (True, True), (False, None)],
    ids=["reparam", "reparam-False", "reparam-True", "nonreparam"],
)
@pytest.mark.parametrize(
    "Elbo",
    [
        Trace_ELBO,
        TraceEnum_ELBO,
    ],
)
def test_particle_gradient(Elbo, reparameterized, has_rsample):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model():
        with pyro.plate("data", len(data)) as ind:
            x = data[ind]
            z = pyro.sample("z", Normal(0, 1))
            pyro.sample("x", Normal(z, 1), obs=x)

    def guide():
        scale = pyro.param("scale", lambda: torch.tensor([1.0]))
        with pyro.plate("data", len(data)):
            loc = pyro.param("loc", lambda: torch.zeros(len(data)), event_dim=0)
            z_dist = Normal(loc, scale)
            if has_rsample is not None:
                z_dist.has_rsample_(has_rsample)
            pyro.sample("z", z_dist)

    elbo = Elbo(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
        num_particles=1,
        strict_enumeration_warning=False,
    )

    # Elbo gradient estimator
    pyro.set_rng_seed(0)
    elbo.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}

    # capture sample values and log_probs
    pyro.set_rng_seed(0)
    guide_tr = poutine.trace(guide).get_trace()
    model_tr = poutine.trace(poutine.replay(model, guide_tr)).get_trace()
    guide_tr.compute_log_prob()
    model_tr.compute_log_prob()
    x = data
    z = guide_tr.nodes["z"]["value"].data
    loc = pyro.param("loc").data
    scale = pyro.param("scale").data

    # expected grads
    if reparameterized and has_rsample is not False:
        # pathwise gradient estimator
        expected_grads = {
            "scale": (
                -(-z * (z - loc) + (x - z) * (z - loc) + 1).sum(0, keepdim=True) / scale
            ),
            "loc": -(-z + (x - z)),
        }
    else:
        # score function gradient estimator
        elbo = (
            model_tr.nodes["x"]["log_prob"].data
            + model_tr.nodes["z"]["log_prob"].data
            - guide_tr.nodes["z"]["log_prob"].data
        )
        dlogq_dloc = (z - loc) / scale**2
        dlogq_dscale = (z - loc) ** 2 / scale**3 - 1 / scale
        if Elbo is TraceEnum_ELBO:
            expected_grads = {
                "scale": -(dlogq_dscale * elbo - dlogq_dscale).sum(0, keepdim=True),
                "loc": -(dlogq_dloc * elbo - dlogq_dloc),
            }
        elif Elbo is Trace_ELBO:
            # expected value of dlogq_dscale and dlogq_dloc is zero
            expected_grads = {
                "scale": -(dlogq_dscale * elbo).sum(0, keepdim=True),
                "loc": -(dlogq_dloc * elbo),
            }

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=1e-4)


@pytest.mark.parametrize("scale", [1.0, 2.0], ids=["unscaled", "scaled"])
@pytest.mark.parametrize(
    "reparameterized,has_rsample",
    [(True, None), (True, False), (True, True), (False, None)],
    ids=["reparam", "reparam-False", "reparam-True", "nonreparam"],
)
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize(
    "Elbo,local_samples",
    [
        (Trace_ELBO, False),
        (DiffTrace_ELBO, False),
        (TraceGraph_ELBO, False),
        (TraceMeanField_ELBO, False),
        (TraceEnum_ELBO, False),
        (TraceEnum_ELBO, True),
    ],
)
def test_subsample_gradient(
    Elbo, reparameterized, has_rsample, subsample, local_samples, scale
):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    subsample_size = 1 if subsample else len(data)
    precision = 0.06 * scale
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model(subsample):
        with pyro.plate("data", len(data), subsample_size, subsample) as ind:
            x = data[ind]
            z = pyro.sample("z", Normal(0, 1))
            pyro.sample("x", Normal(z, 1), obs=x)

    def guide(subsample):
        scale = pyro.param("scale", lambda: torch.tensor([1.0]))
        with pyro.plate("data", len(data), subsample_size, subsample):
            loc = pyro.param("loc", lambda: torch.zeros(len(data)), event_dim=0)
            z_dist = Normal(loc, scale)
            if has_rsample is not None:
                z_dist.has_rsample_(has_rsample)
            pyro.sample("z", z_dist)

    if scale != 1.0:
        model = poutine.scale(model, scale=scale)
        guide = poutine.scale(guide, scale=scale)

    num_particles = 50000
    if local_samples:
        guide = config_enumerate(guide, num_samples=num_particles)
        num_particles = 1

    optim = Adam({"lr": 0.1})
    elbo = Elbo(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
        num_particles=num_particles,
        vectorize_particles=True,
        strict_enumeration_warning=False,
    )
    inference = SVI(model, guide, optim, loss=elbo)
    with xfail_if_not_implemented():
        if subsample_size == 1:
            inference.loss_and_grads(
                model, guide, subsample=torch.tensor([0], dtype=torch.long)
            )
            inference.loss_and_grads(
                model, guide, subsample=torch.tensor([1], dtype=torch.long)
            )
        else:
            inference.loss_and_grads(
                model, guide, subsample=torch.tensor([0, 1], dtype=torch.long)
            )
    params = dict(pyro.get_param_store().named_parameters())
    normalizer = 2 if subsample else 1
    actual_grads = {
        name: param.grad.detach().cpu().numpy() / normalizer
        for name, param in params.items()
    }

    expected_grads = {
        "loc": scale * np.array([0.5, -2.0]),
        "scale": scale * np.array([2.0]),
    }
    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize(
    "reparameterized", [True, False], ids=["reparam", "nonreparam"]
)
@pytest.mark.parametrize(
    "Elbo", [Trace_ELBO, DiffTrace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO]
)
def test_plate(Elbo, reparameterized):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    num_particles = 200000
    precision = 0.06
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model():
        particles_plate = pyro.plate("particles", num_particles, dim=-2)
        data_plate = pyro.plate("data", len(data), dim=-1)

        pyro.sample("nuisance_a", Normal(0, 1))
        with particles_plate, data_plate:
            z = pyro.sample("z", Normal(0, 1))
        pyro.sample("nuisance_b", Normal(2, 3))
        with data_plate, particles_plate:
            pyro.sample("x", Normal(z, 1), obs=data)
        pyro.sample("nuisance_c", Normal(4, 5))

    def guide():
        loc = pyro.param("loc", torch.zeros(len(data)))
        scale = pyro.param("scale", torch.tensor([1.0]))

        pyro.sample("nuisance_c", Normal(4, 5))
        with pyro.plate("particles", num_particles, dim=-2):
            with pyro.plate("data", len(data), dim=-1):
                pyro.sample("z", Normal(loc, scale))
        pyro.sample("nuisance_b", Normal(2, 3))
        pyro.sample("nuisance_a", Normal(0, 1))

    optim = Adam({"lr": 0.1})
    elbo = Elbo(strict_enumeration_warning=False)
    inference = SVI(model, guide, optim, loss=elbo)
    inference.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {
        name: param.grad.detach().cpu().numpy() / num_particles
        for name, param in params.items()
    }

    expected_grads = {"loc": np.array([0.5, -2.0]), "scale": np.array([2.0])}
    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize(
    "reparameterized", [True, False], ids=["reparam", "nonreparam"]
)
@pytest.mark.parametrize(
    "Elbo", [Trace_ELBO, DiffTrace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO]
)
def test_plate_elbo_vectorized_particles(Elbo, reparameterized):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    num_particles = 200000
    precision = 0.06
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model():
        data_plate = pyro.plate("data", len(data))

        pyro.sample("nuisance_a", Normal(0, 1))
        with data_plate:
            z = pyro.sample("z", Normal(0, 1))
        pyro.sample("nuisance_b", Normal(2, 3))
        with data_plate:
            pyro.sample("x", Normal(z, 1), obs=data)
        pyro.sample("nuisance_c", Normal(4, 5))

    def guide():
        loc = pyro.param("loc", torch.zeros(len(data)))
        scale = pyro.param("scale", torch.tensor([1.0]))

        pyro.sample("nuisance_c", Normal(4, 5))
        with pyro.plate("data", len(data)):
            pyro.sample("z", Normal(loc, scale))
        pyro.sample("nuisance_b", Normal(2, 3))
        pyro.sample("nuisance_a", Normal(0, 1))

    optim = Adam({"lr": 0.1})
    loss = Elbo(
        num_particles=num_particles,
        vectorize_particles=True,
        strict_enumeration_warning=False,
    )
    inference = SVI(model, guide, optim, loss=loss)
    inference.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {
        name: param.grad.detach().cpu().numpy() for name, param in params.items()
    }

    expected_grads = {"loc": np.array([0.5, -2.0]), "scale": np.array([2.0])}
    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize(
    "reparameterized", [True, False], ids=["reparam", "nonreparam"]
)
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize(
    "Elbo",
    [
        Trace_ELBO,
        TraceGraph_ELBO,
        TraceEnum_ELBO,
        TraceMeanField_ELBO,
        xfail_param(
            JitTrace_ELBO,
            reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor",
        ),
        xfail_param(
            JitTraceGraph_ELBO,
            reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor",
        ),
        xfail_param(
            JitTraceEnum_ELBO,
            reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor",
        ),
        xfail_param(
            JitTraceMeanField_ELBO,
            reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor",
        ),
    ],
)
def test_subsample_gradient_sequential(Elbo, reparameterized, subsample):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    subsample_size = 1 if subsample else len(data)
    num_particles = 5000
    precision = 0.333
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model():
        with pyro.plate("data", len(data), subsample_size) as ind:
            x = data[ind]
            z = pyro.sample("z", Normal(0, 1).expand_by(x.shape))
            pyro.sample("x", Normal(z, 1), obs=x)

    def guide():
        loc = pyro.param("loc", lambda: torch.zeros(len(data), requires_grad=True))
        scale = pyro.param("scale", lambda: torch.tensor([1.0], requires_grad=True))
        with pyro.plate("data", len(data), subsample_size) as ind:
            pyro.sample("z", Normal(loc[ind], scale))

    optim = Adam({"lr": 0.1})
    elbo = Elbo(num_particles=10, strict_enumeration_warning=False)
    inference = SVI(model, guide, optim, elbo)
    iters = num_particles // 10
    with xfail_if_not_implemented():
        for _ in range(iters):
            inference.loss_and_grads(model, guide)

    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {
        name: param.grad.detach().cpu().numpy() / iters
        for name, param in params.items()
    }

    expected_grads = {"loc": np.array([0.5, -2.0]), "scale": np.array([2.0])}
    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.stage("funsor")
def test_collapse_beta_binomial():
    pytest.importorskip("funsor")

    total_count = 10
    data = torch.tensor(3.0)

    def model1():
        c1 = pyro.param("c1", torch.tensor(0.5), constraint=constraints.positive)
        c0 = pyro.param("c0", torch.tensor(1.5), constraint=constraints.positive)
        with poutine.collapse():
            probs = pyro.sample("probs", dist.Beta(c1, c0))
            pyro.sample("obs", dist.Binomial(total_count, probs), obs=data)

    def model2():
        c1 = pyro.param("c1", torch.tensor(0.5), constraint=constraints.positive)
        c0 = pyro.param("c0", torch.tensor(1.5), constraint=constraints.positive)
        pyro.sample("obs", dist.BetaBinomial(c1, c0, total_count), obs=data)

    trace1 = poutine.trace(model1).get_trace()
    trace2 = poutine.trace(model2).get_trace()
    assert "probs" in trace1.nodes
    assert "obs" not in trace1.nodes
    assert "probs" not in trace2.nodes
    assert "obs" in trace2.nodes

    logp1 = trace1.log_prob_sum()
    logp2 = trace2.log_prob_sum()
    assert_equal(logp1.detach().item(), logp2.detach().item())

    log_c1 = pyro.param("c1").unconstrained()
    log_c0 = pyro.param("c0").unconstrained()
    grads1 = torch.autograd.grad(logp1, [log_c1, log_c0], retain_graph=True)
    grads2 = torch.autograd.grad(logp2, [log_c1, log_c0], retain_graph=True)
    for g1, g2, name in zip(grads1, grads2, ["log(c1)", "log(c0)"]):
        print(g1, g2)
        assert_equal(g1, g2, msg=name)
