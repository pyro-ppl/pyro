from __future__ import absolute_import, division, print_function

import pytest
import torch
import torch.optim
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.util import ng_ones, ng_zeros, zero_grads
from tests.common import assert_equal


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["Trace", "TraceGraph"])
def test_subsample_gradient(trace_graph, reparameterized):
    pyro.clear_param_store()
    data_size = 2
    subsample_size = 1
    num_particles = 1000
    precision = 0.333
    data = dist.normal(ng_zeros(data_size), ng_ones(data_size))

    def model(subsample_size):
        with pyro.iarange("data", len(data), subsample_size) as ind:
            x = data[ind]
            z = pyro.sample("z", dist.Normal(ng_zeros(len(x)), ng_ones(len(x)),
                                             reparameterized=reparameterized))
            pyro.observe("x", dist.Normal(z, ng_ones(len(x)), reparameterized=reparameterized), x)

    def guide(subsample_size):
        mu = pyro.param("mu", lambda: Variable(torch.zeros(len(data)), requires_grad=True))
        sigma = pyro.param("sigma", lambda: Variable(torch.ones(1), requires_grad=True))
        with pyro.iarange("data", len(data), subsample_size) as ind:
            mu = mu[ind]
            sigma = sigma.expand(subsample_size)
            pyro.sample("z", dist.Normal(mu, sigma, reparameterized=reparameterized))

    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO",
                    trace_graph=trace_graph, num_particles=num_particles)

    # Compute gradients without subsampling.
    inference.loss_and_grads(model, guide, subsample_size=data_size)
    params = dict(pyro.get_param_store().named_parameters())
    expected_grads = {name: param.grad.data.clone() for name, param in params.items()}
    zero_grads(params.values())

    # Compute gradients with subsampling.
    inference.loss_and_grads(model, guide, subsample_size=subsample_size)
    actual_grads = {name: param.grad.data.clone() for name, param in params.items()}

    for name in sorted(params):
        print('\nexpected {} = {}'.format(name, expected_grads[name].cpu().numpy()))
        print('actual   {} = {}'.format(name, actual_grads[name].cpu().numpy()))
    assert_equal(actual_grads, expected_grads, prec=precision)
