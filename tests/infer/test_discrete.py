from __future__ import absolute_import, division, print_function

import logging

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.discrete import infer_discrete
from pyro.infer.enum import config_enumerate
from tests.common import assert_equal

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('temperature', [0, 1], ids=['map', 'sample'])
def test_sample_posterior_1(temperature):
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = torch.tensor([1., 2., 3.])

    @config_enumerate(default="parallel")
    def model(num_particles=1, z=None):
        p = pyro.param("p", torch.tensor(0.25))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z = pyro.sample("z", dist.Bernoulli(p), obs=z)
            logger.info("z.shape = {}".format(z.shape))
            with pyro.plate("data", 3):
                pyro.sample("x", dist.Normal(z, 1.), obs=data)

    first_available_dim = -3
    sampled_model = infer_discrete(model, first_available_dim, temperature)
    sampled_trace = poutine.trace(sampled_model).get_trace(num_particles)
    conditioned_traces = {z: poutine.trace(model).get_trace(z=torch.tensor(z)) for z in [0., 1.]}

    # Check  posterior over z.
    actual_z_mean = sampled_trace.nodes["z"]["value"].mean()
    if temperature:
        expected_z_mean = 1 / (1 + (conditioned_traces[0].log_prob_sum() -
                                    conditioned_traces[1].log_prob_sum()).exp())
    else:
        expected_z_mean = (conditioned_traces[1].log_prob_sum() >
                           conditioned_traces[0].log_prob_sum()).float()
    assert_equal(actual_z_mean, expected_z_mean, prec=1e-2)


@pytest.mark.parametrize('temperature', [0, 1], ids=['map', 'sample'])
def test_sample_posterior_2(temperature):
    #       +--------+
    #  z1 --|--> x1  |
    #   |   |        |
    #   V   |        |
    #  z2 --|--> x2  |
    #       +--------+
    num_particles = 10000
    data = torch.tensor([[-1., -1., 0.], [-1., 1., 1.]])

    @config_enumerate(default="parallel")
    def model(num_particles=1, z1=None, z2=None):
        p = pyro.param("p", torch.tensor([[0.25, 0.75], [0.1, 0.9]]))
        loc = pyro.param("loc", torch.tensor([-1., 1.]))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z1 = pyro.sample("z1", dist.Categorical(p[0]), obs=z1)
            z2 = pyro.sample("z2", dist.Categorical(p[z1]), obs=z2)
            logger.info("z1.shape = {}".format(z1.shape))
            logger.info("z2.shape = {}".format(z2.shape))
            with pyro.plate("data", 3):
                pyro.sample("x1", dist.Normal(loc[z1], 1.), obs=data[0])
                pyro.sample("x2", dist.Normal(loc[z2], 1.), obs=data[1])

    first_available_dim = -3
    sampled_model = infer_discrete(model, first_available_dim, temperature)
    sampled_trace = poutine.trace(
        sampled_model).get_trace(num_particles)
    conditioned_traces = {(z1, z2): poutine.trace(model).get_trace(z1=torch.tensor(z1),
                                                                   z2=torch.tensor(z2))
                          for z1 in [0, 1] for z2 in [0, 1]}

    # Check joint posterior over (z1, z2).
    actual_probs = torch.empty(2, 2)
    expected_probs = torch.empty(2, 2)
    for (z1, z2), tr in conditioned_traces.items():
        expected_probs[z1, z2] = tr.log_prob_sum().exp()
        actual_probs[z1, z2] = ((sampled_trace.nodes["z1"]["value"] == z1) &
                                (sampled_trace.nodes["z2"]["value"] == z2)).float().mean()
    if temperature:
        expected_probs = expected_probs / expected_probs.sum()
    else:
        argmax = expected_probs.reshape(-1).max(0)[1]
        expected_probs[:] = 0
        expected_probs.reshape(-1)[argmax] = 1
    assert_equal(expected_probs, actual_probs, prec=1e-2)


@pytest.mark.parametrize('temperature', [0, 1], ids=['map', 'sample'])
def test_sample_posterior_3(temperature):
    #       +---------+  +---------------+
    #  z1 --|--> x1   |  |  z2 ---> x2   |
    #       |       3 |  |             2 |
    #       +---------+  +---------------+
    num_particles = 10000
    data = [torch.tensor([-1., -1., 0.]), torch.tensor([-1., 1.])]

    @config_enumerate(default="parallel")
    def model(num_particles=1, z1=None, z2=None):
        p = pyro.param("p", torch.tensor([0.25, 0.75]))
        loc = pyro.param("loc", torch.tensor([-1., 1.]))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z1 = pyro.sample("z1", dist.Categorical(p), obs=z1)
            with pyro.plate("data[0]", 3):
                pyro.sample("x1", dist.Normal(loc[z1], 1.), obs=data[0])
            with pyro.plate("data[1]", 2):
                z2 = pyro.sample("z2", dist.Categorical(p), obs=z2)
                pyro.sample("x2", dist.Normal(loc[z2], 1.), obs=data[1])

    first_available_dim = -3
    sampled_model = infer_discrete(model, first_available_dim, temperature)
    sampled_trace = poutine.trace(
        sampled_model).get_trace(num_particles)
    conditioned_traces = {(z1, z20, z21): poutine.trace(model).get_trace(z1=torch.tensor(z1),
                                                                         z2=torch.tensor([z20, z21]))
                          for z1 in [0, 1] for z20 in [0, 1] for z21 in [0, 1]}

    # Check joint posterior over (z1, z2[0], z2[1]).
    actual_probs = torch.empty(2, 2, 2)
    expected_probs = torch.empty(2, 2, 2)
    for (z1, z20, z21), tr in conditioned_traces.items():
        expected_probs[z1, z20, z21] = tr.log_prob_sum().exp()
        actual_probs[z1, z20, z21] = ((sampled_trace.nodes["z1"]["value"] == z1) &
                                      (sampled_trace.nodes["z2"]["value"][..., :1] == z20) &
                                      (sampled_trace.nodes["z2"]["value"][..., 1:] == z21)).float().mean()
    if temperature:
        expected_probs = expected_probs / expected_probs.sum()
    else:
        argmax = expected_probs.reshape(-1).max(0)[1]
        expected_probs[:] = 0
        expected_probs.reshape(-1)[argmax] = 1
    assert_equal(expected_probs.reshape(-1), actual_probs.reshape(-1), prec=1e-2)
