# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest
import torch
import torch.optim

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.testing import fakes
from pyro.infer import (SVI, JitTrace_ELBO, JitTraceEnum_ELBO, JitTraceGraph_ELBO, JitTraceMeanField_ELBO, Trace_ELBO,
                        TraceEnum_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO, config_enumerate)
from pyro.optim import Adam
from tests.common import assert_equal, xfail_if_not_implemented, xfail_param

logger = logging.getLogger(__name__)


def DiffTrace_ELBO(*args, **kwargs):
    return Trace_ELBO(*args, **kwargs).differentiable_loss


@pytest.mark.parametrize("scale", [1., 2.], ids=["unscaled", "scaled"])
@pytest.mark.parametrize("reparameterized,has_rsample",
                         [(True, None), (True, False), (True, True), (False, None)],
                         ids=["reparam", "reparam-False", "reparam-True", "nonreparam"])
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo,local_samples", [
    (Trace_ELBO, False),
    (DiffTrace_ELBO, False),
    (TraceGraph_ELBO, False),
    (TraceMeanField_ELBO, False),
    (TraceEnum_ELBO, False),
    (TraceEnum_ELBO, True),
])
def test_subsample_gradient(Elbo, reparameterized, has_rsample, subsample, local_samples, scale):
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
    elbo = Elbo(max_plate_nesting=1,  # set this to ensure rng agrees across runs
                num_particles=num_particles,
                vectorize_particles=True,
                strict_enumeration_warning=False)
    inference = SVI(model, guide, optim, loss=elbo)
    with xfail_if_not_implemented():
        if subsample_size == 1:
            inference.loss_and_grads(model, guide, subsample=torch.tensor([0], dtype=torch.long))
            inference.loss_and_grads(model, guide, subsample=torch.tensor([1], dtype=torch.long))
        else:
            inference.loss_and_grads(model, guide, subsample=torch.tensor([0, 1], dtype=torch.long))
    params = dict(pyro.get_param_store().named_parameters())
    normalizer = 2 if subsample else 1
    actual_grads = {name: param.grad.detach().cpu().numpy() / normalizer for name, param in params.items()}

    expected_grads = {'loc': scale * np.array([0.5, -2.0]), 'scale': scale * np.array([2.0])}
    for name in sorted(params):
        logger.info('expected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, DiffTrace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
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
        scale = pyro.param("scale", torch.tensor([1.]))

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
    actual_grads = {name: param.grad.detach().cpu().numpy() / num_particles
                    for name, param in params.items()}

    expected_grads = {'loc': np.array([0.5, -2.0]), 'scale': np.array([2.0])}
    for name in sorted(params):
        logger.info('expected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, DiffTrace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
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
        scale = pyro.param("scale", torch.tensor([1.]))

        pyro.sample("nuisance_c", Normal(4, 5))
        with pyro.plate("data", len(data)):
            pyro.sample("z", Normal(loc, scale))
        pyro.sample("nuisance_b", Normal(2, 3))
        pyro.sample("nuisance_a", Normal(0, 1))

    optim = Adam({"lr": 0.1})
    loss = Elbo(num_particles=num_particles,
                vectorize_particles=True,
                strict_enumeration_warning=False)
    inference = SVI(model, guide, optim, loss=loss)
    inference.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.detach().cpu().numpy()
                    for name, param in params.items()}

    expected_grads = {'loc': np.array([0.5, -2.0]), 'scale': np.array([2.0])}
    for name in sorted(params):
        logger.info('expected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceEnum_ELBO,
    TraceMeanField_ELBO,
    xfail_param(JitTrace_ELBO,
                reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor"),
    xfail_param(JitTraceGraph_ELBO,
                reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor"),
    xfail_param(JitTraceEnum_ELBO,
                reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor"),
    xfail_param(JitTraceMeanField_ELBO,
                reason="in broadcast_all: RuntimeError: expected int at position 0, but got: Tensor"),
])
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
    actual_grads = {name: param.grad.detach().cpu().numpy() / iters
                    for name, param in params.items()}

    expected_grads = {'loc': np.array([0.5, -2.0]), 'scale': np.array([2.0])}
    for name in sorted(params):
        logger.info('expected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)
