from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pytest
import torch
import torch.optim
from torch.autograd import variable

import pyro
import pyro.distributions as dist
from pyro.distributions.testing import fakes
from pyro.infer import SVI
from pyro.optim import Adam
from tests.common import assert_equal

logger = logging.getLogger(__name__)


# TODO make this pass for trace_elbo
@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(True, False), (False, True)],
                         ids=["TraceGraph", "TraceEnum"])
def test_subsample_gradient_parallelized(trace_graph, enum_discrete, reparameterized, subsample):
    pyro.clear_param_store()
    data = variable([-0.5, 2.0])
    subsample_size = 1 if subsample else len(data)
    num_particles = 20000
    precision = 0.05
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model(subsample):
        with pyro.iarange("particles", num_particles):
            with pyro.iarange("data", len(data), subsample_size, subsample) as ind:
                x = data[ind].unsqueeze(-1).expand(-1, num_particles)
                z = pyro.sample("z", Normal(0, 1).reshape(x.shape))
                pyro.sample("x", Normal(z, 1), obs=x)

    def guide(subsample):
        mu = pyro.param("mu", lambda: variable(torch.zeros(len(data)), requires_grad=True))
        sigma = pyro.param("sigma", lambda: variable([1.0], requires_grad=True))
        with pyro.iarange("particles", num_particles):
            with pyro.iarange("data", len(data), subsample_size, subsample) as ind:
                mu_ind = mu[ind].unsqueeze(-1).expand(-1, num_particles)
                pyro.sample("z", Normal(mu_ind, sigma))

    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO",
                    trace_graph=trace_graph, enum_discrete=enum_discrete,
                    num_particles=1)
    if subsample_size == 1:
        inference.loss_and_grads(model, guide, subsample=torch.LongTensor([0]))
        inference.loss_and_grads(model, guide, subsample=torch.LongTensor([1]))
    else:
        inference.loss_and_grads(model, guide, subsample=torch.LongTensor([0, 1]))
    params = dict(pyro.get_param_store().named_parameters())
    normalizer = 2 * num_particles / subsample_size
    actual_grads = {name: param.grad.detach().cpu().numpy() / normalizer for name, param in params.items()}

    expected_grads = {'mu': np.array([0.5, -2.0]), 'sigma': np.array([2.0])}
    for name in sorted(params):
        logger.info('expected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
def test_subsample_gradient(trace_graph, enum_discrete, reparameterized, subsample):
    pyro.clear_param_store()
    data = variable([-0.5, 2.0])
    subsample_size = 1 if subsample else len(data)
    num_particles = 5000
    precision = 0.333
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model():
        with pyro.iarange("data", len(data), subsample_size) as ind:
            x = data[ind]
            z = pyro.sample("z", Normal(0, 1).reshape(x.shape))
            pyro.sample("x", Normal(z, 1), obs=x)

    def guide():
        mu = pyro.param("mu", lambda: variable(torch.zeros(len(data)), requires_grad=True))
        sigma = pyro.param("sigma", lambda: variable([1.0], requires_grad=True))
        with pyro.iarange("data", len(data), subsample_size) as ind:
            pyro.sample("z", Normal(mu[ind], sigma))

    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO",
                    trace_graph=trace_graph, enum_discrete=enum_discrete,
                    num_particles=num_particles)
    inference.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.detach().cpu().numpy() for name, param in params.items()}

    expected_grads = {'mu': np.array([0.5, -2.0]), 'sigma': np.array([2.0])}
    for name in sorted(params):
        logger.info('expected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)
