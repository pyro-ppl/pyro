from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pytest
import torch
import torch.optim
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.distributions.testing import fakes
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("reparameterized", [True, False], ids=["reparam", "nonreparam"])
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize("trace_graph", [False, True], ids=["Trace", "TraceGraph"])
def test_subsample_gradient(trace_graph, reparameterized, subsample):
    pyro.clear_param_store()
    data = Variable(torch.Tensor([-0.5, 2.0]))
    subsample_size = 1 if subsample else len(data)
    num_particles = 5000
    precision = 0.333
    normal = dist.normal if reparameterized else fakes.nonreparameterized_normal

    def model():
        with pyro.iarange("data", len(data), subsample_size) as ind:
            x = data[ind]
            z = pyro.sample("z", normal, ng_zeros(len(x)), ng_ones(len(x)))
            pyro.sample("x", normal, z, ng_ones(len(x)), obs=x)

    def guide():
        mu = pyro.param("mu", lambda: Variable(torch.zeros(len(data)), requires_grad=True))
        sigma = pyro.param("sigma", lambda: Variable(torch.ones(1), requires_grad=True))
        with pyro.iarange("data", len(data), subsample_size) as ind:
            mu = mu[ind]
            sigma = sigma.expand(subsample_size)
            pyro.sample("z", normal, mu, sigma)

    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO",
                    trace_graph=trace_graph, num_particles=num_particles)
    inference.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.data.cpu().numpy() for name, param in params.items()}

    expected_grads = {'mu': np.array([0.5, -2.0]), 'sigma': np.array([2.0])}
    for name in sorted(params):
        logger.info('\nexpected {} = {}'.format(name, expected_grads[name]))
        logger.info('actual   {} = {}'.format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)
