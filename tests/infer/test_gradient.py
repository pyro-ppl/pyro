from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pytest
import torch
import torch.optim

import pyro
import pyro.distributions as dist
from pyro.distributions.testing import fakes
from pyro.infer import (SVI, JitTrace_ELBO, JitTraceEnum_ELBO, JitTraceGraph_ELBO, Trace_ELBO, TraceEnum_ELBO,
                        TraceGraph_ELBO)
from pyro.optim import Adam
import pyro.poutine as poutine
from tests.common import assert_equal, xfail_param

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_subsample_gradient(Elbo, reparameterized, subsample):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    subsample_size = 1 if subsample else len(data)
    precision = 0.06
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    @poutine.broadcast
    def model(subsample):
        with pyro.iarange("data", len(data), subsample_size, subsample) as ind:
            x = data[ind]
            z = pyro.sample("z", Normal(0, 1))
            pyro.sample("x", Normal(z, 1), obs=x)

    @poutine.broadcast
    def guide(subsample):
        loc = pyro.param("loc", lambda: torch.zeros(len(data), requires_grad=True))
        scale = pyro.param("scale", lambda: torch.tensor([1.0], requires_grad=True))
        with pyro.iarange("data", len(data), subsample_size, subsample) as ind:
            loc_ind = loc[ind]
            pyro.sample("z", Normal(loc_ind, scale))

    optim = Adam({"lr": 0.1})
    elbo = Elbo(max_iarange_nesting=1,
                num_particles=50000,
                vectorize_particles=True,
                strict_enumeration_warning=False)
    inference = SVI(model, guide, optim, loss=elbo)
    if subsample_size == 1:
        inference.loss_and_grads(model, guide, subsample=torch.LongTensor([0]))
        inference.loss_and_grads(model, guide, subsample=torch.LongTensor([1]))
    else:
        inference.loss_and_grads(model, guide, subsample=torch.LongTensor([0, 1]))
    params = dict(pyro.get_param_store().named_parameters())
    normalizer = 2 if subsample else 1
    actual_grads = {name: param.grad.detach().cpu().numpy() / normalizer for name, param in params.items()}

    expected_grads = {'loc': np.array([0.5, -2.0]), 'scale': np.array([2.0])}
    for name in sorted(params):
        logger.info('expected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange(Elbo, reparameterized):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    num_particles = 100000
    precision = 0.06
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    @poutine.broadcast
    def model():
        particles_iarange = pyro.iarange("particles", num_particles, dim=-2)
        data_iarange = pyro.iarange("data", len(data), dim=-1)

        pyro.sample("nuisance_a", Normal(0, 1))
        with particles_iarange, data_iarange:
            z = pyro.sample("z", Normal(0, 1))
        pyro.sample("nuisance_b", Normal(2, 3))
        with data_iarange, particles_iarange:
            pyro.sample("x", Normal(z, 1), obs=data)
        pyro.sample("nuisance_c", Normal(4, 5))

    @poutine.broadcast
    def guide():
        loc = pyro.param("loc", torch.zeros(len(data)))
        scale = pyro.param("scale", torch.tensor([1.]))

        pyro.sample("nuisance_c", Normal(4, 5))
        with pyro.iarange("particles", num_particles, dim=-2):
            with pyro.iarange("data", len(data), dim=-1):
                pyro.sample("z", Normal(loc, scale))
        pyro.sample("nuisance_b", Normal(2, 3))
        pyro.sample("nuisance_a", Normal(0, 1))

    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss=Elbo(strict_enumeration_warning=False))
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
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_elbo_vectorized_particles(Elbo, reparameterized):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    num_particles = 200000
    precision = 0.06
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    @poutine.broadcast
    def model():
        data_iarange = pyro.iarange("data", len(data))

        pyro.sample("nuisance_a", Normal(0, 1))
        with data_iarange:
            z = pyro.sample("z", Normal(0, 1))
        pyro.sample("nuisance_b", Normal(2, 3))
        with data_iarange:
            pyro.sample("x", Normal(z, 1), obs=data)
        pyro.sample("nuisance_c", Normal(4, 5))

    @poutine.broadcast
    def guide():
        loc = pyro.param("loc", torch.zeros(len(data)))
        scale = pyro.param("scale", torch.tensor([1.]))

        pyro.sample("nuisance_c", Normal(4, 5))
        with pyro.iarange("data", len(data)):
            pyro.sample("z", Normal(loc, scale))
        pyro.sample("nuisance_b", Normal(2, 3))
        pyro.sample("nuisance_a", Normal(0, 1))

    optim = Adam({"lr": 0.1})
    loss = Elbo(max_iarange_nesting=1,
                num_particles=num_particles,
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
    xfail_param(JitTrace_ELBO,
                reason="jit RuntimeError: Unsupported op descriptor: index-2"),
    xfail_param(JitTraceGraph_ELBO,
                reason="jit RuntimeError: Unsupported op descriptor: index-2"),
    xfail_param(JitTraceEnum_ELBO,
                reason="jit RuntimeError: Unsupported op descriptor: index-2"),
])
def test_subsample_gradient_sequential(Elbo, reparameterized, subsample):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    subsample_size = 1 if subsample else len(data)
    num_particles = 5000
    precision = 0.333
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model():
        with pyro.iarange("data", len(data), subsample_size) as ind:
            x = data[ind]
            z = pyro.sample("z", Normal(0, 1).expand_by(x.shape))
            pyro.sample("x", Normal(z, 1), obs=x)

    def guide():
        loc = pyro.param("loc", lambda: torch.zeros(len(data), requires_grad=True))
        scale = pyro.param("scale", lambda: torch.tensor([1.0], requires_grad=True))
        with pyro.iarange("data", len(data), subsample_size) as ind:
            pyro.sample("z", Normal(loc[ind], scale))

    optim = Adam({"lr": 0.1})
    elbo = Elbo(num_particles=10, strict_enumeration_warning=False)
    inference = SVI(model, guide, optim, elbo)
    iters = num_particles // 10
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
