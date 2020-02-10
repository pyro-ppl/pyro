# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import TraceEnum_ELBO
from pyro.infer.discrete import TraceEnumSample_ELBO, infer_discrete
from pyro.infer.enum import config_enumerate
from tests.common import assert_equal

logger = logging.getLogger(__name__)


def elbo_infer_discrete(model, first_available_dim, temperature):
    """
    Wrapper around ``TraceEnumSample_ELBO`` to test agreement with
    ``TraceEnum_ELBO`` and then return ``.sample_saved()``.
    """
    assert temperature == 1
    max_plate_nesting = -first_available_dim - 1
    expected_elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
    actual_elbo = TraceEnumSample_ELBO(max_plate_nesting=max_plate_nesting)

    def empty_guide(*args, **kwargs):
        pass

    def inferred_model(*args, **kwargs):
        with poutine.block():
            expected_loss = expected_elbo.loss(model, empty_guide, *args, **kwargs)
            actual_loss = actual_elbo.loss(model, empty_guide, *args, **kwargs)
        assert_equal(actual_loss, expected_loss)
        return actual_elbo.sample_saved()

    return inferred_model


def log_mean_prob(trace, particle_dim):
    """
    Marginalizes out particle_dim from a trace.
    """
    assert particle_dim < 0
    trace.compute_log_prob()
    total = 0.
    for node in trace.nodes.values():
        if node["type"] == "sample" and type(node["fn"]).__name__ != "_Subsample":
            log_prob = node["log_prob"]
            assert log_prob.dim() == -particle_dim
            num_particles = log_prob.size(0)
            total = total + log_prob.reshape(num_particles, -1).sum(-1)
    return total.logsumexp(0) - math.log(num_particles)


@pytest.mark.parametrize('infer,temperature', [
    (infer_discrete, 0),
    (infer_discrete, 1),
    (elbo_infer_discrete, 1),
], ids=['map', 'sample', 'sample-elbo'])
@pytest.mark.parametrize('plate_size', [2])
def test_plate_smoke(infer, temperature, plate_size):
    #       +-----------------+
    #  z1 --|--> z2 ---> x2   |
    #       |               N | for N in {1,2}
    #       +-----------------+

    @config_enumerate
    def model():
        p = pyro.param("p", torch.tensor([0.25, 0.75]))
        q = pyro.param("q", torch.tensor([[0.25, 0.75], [0.75, 0.25]]))
        loc = pyro.param("loc", torch.tensor([-1., 1.]))
        z1 = pyro.sample("z1", dist.Categorical(p))
        with pyro.plate("plate", plate_size):
            z2 = pyro.sample("z2", dist.Categorical(q[z1]))
            pyro.sample("x2", dist.Normal(loc[z2], 1.), obs=torch.ones(plate_size))

    first_available_dim = -2
    infer(model, first_available_dim, temperature)()


@pytest.mark.parametrize('infer,temperature', [
    (infer_discrete, 0),
    (infer_discrete, 1),
    (elbo_infer_discrete, 1),
], ids=['map', 'sample', 'sample-elbo'])
def test_distribution_1(infer, temperature):
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = torch.tensor([1., 2., 3.])

    @config_enumerate
    def model(num_particles=1, z=None):
        p = pyro.param("p", torch.tensor(0.25))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z = pyro.sample("z", dist.Bernoulli(p), obs=z)
            logger.info("z.shape = {}".format(z.shape))
            with pyro.plate("data", 3):
                pyro.sample("x", dist.Normal(z, 1.), obs=data)

    first_available_dim = -3
    sampled_model = infer(model, first_available_dim, temperature)
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


@pytest.mark.parametrize('infer,temperature', [
    (infer_discrete, 0),
    (infer_discrete, 1),
    (elbo_infer_discrete, 1),
], ids=['map', 'sample', 'sample-elbo'])
def test_distribution_2(infer, temperature):
    #       +--------+
    #  z1 --|--> x1  |
    #   |   |        |
    #   V   |        |
    #  z2 --|--> x2  |
    #       +--------+
    num_particles = 10000
    data = torch.tensor([[-1., -1., 0.], [-1., 1., 1.]])

    @config_enumerate
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
    sampled_model = infer(model, first_available_dim, temperature)
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


@pytest.mark.parametrize('infer,temperature', [
    (infer_discrete, 0),
    (infer_discrete, 1),
    (elbo_infer_discrete, 1),
], ids=['map', 'sample', 'sample-elbo'])
def test_distribution_3(infer, temperature):
    #       +---------+  +---------------+
    #  z1 --|--> x1   |  |  z2 ---> x2   |
    #       |       3 |  |             2 |
    #       +---------+  +---------------+
    num_particles = 10000
    data = [torch.tensor([-1., -1., 0.]), torch.tensor([-1., 1.])]

    @config_enumerate
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
    sampled_model = infer(model, first_available_dim, temperature)
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


@pytest.mark.parametrize('infer,temperature', [
    (infer_discrete, 0),
    (infer_discrete, 1),
    (elbo_infer_discrete, 1),
], ids=['map', 'sample', 'sample-elbo'])
def test_distribution_masked(infer, temperature):
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = torch.tensor([1., 2., 3.])
    mask = torch.tensor([True, False, False])

    @config_enumerate
    def model(num_particles=1, z=None):
        p = pyro.param("p", torch.tensor(0.25))
        with pyro.plate("num_particles", num_particles, dim=-2):
            z = pyro.sample("z", dist.Bernoulli(p), obs=z)
            logger.info("z.shape = {}".format(z.shape))
            with pyro.plate("data", 3), poutine.mask(mask=mask):
                pyro.sample("x", dist.Normal(z, 1.), obs=data)

    first_available_dim = -3
    sampled_model = infer(model, first_available_dim, temperature)
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


@pytest.mark.parametrize('length', [1, 2, 10, 100])
@pytest.mark.parametrize('infer,temperature', [
    (infer_discrete, 0),
    (infer_discrete, 1),
    (elbo_infer_discrete, 1),
], ids=['map', 'sample', 'sample-elbo'])
def test_hmm_smoke(infer, temperature, length):

    # This should match the example in the infer_discrete docstring.
    def hmm(data, hidden_dim=10):
        transition = 0.3 / hidden_dim + 0.7 * torch.eye(hidden_dim)
        means = torch.arange(float(hidden_dim))
        states = [0]
        for t in pyro.markov(range(len(data))):
            states.append(pyro.sample("states_{}".format(t),
                                      dist.Categorical(transition[states[-1]])))
            data[t] = pyro.sample("obs_{}".format(t),
                                  dist.Normal(means[states[-1]], 1.),
                                  obs=data[t])
        return states, data

    true_states, data = hmm([None] * length)
    assert len(data) == length
    assert len(true_states) == 1 + len(data)

    decoder = infer(config_enumerate(hmm),
                    first_available_dim=-1, temperature=temperature)
    inferred_states, _ = decoder(data)
    assert len(inferred_states) == len(true_states)

    logger.info("true states: {}".format(list(map(int, true_states))))
    logger.info("inferred states: {}".format(list(map(int, inferred_states))))


@pytest.mark.xfail(reason='infer_discrete log_prob is incorrect')
@pytest.mark.parametrize('nderivs', [0, 1], ids=['value', 'grad'])
def test_prob(nderivs):
    #      +-------+
    #  z --|--> x  |
    #      +-------+
    num_particles = 10000
    data = torch.tensor([0.5, 1., 1.5])
    p = pyro.param("p", torch.tensor(0.25))

    @config_enumerate
    def model(num_particles):
        p = pyro.param("p")
        with pyro.plate("num_particles", num_particles, dim=-2):
            z = pyro.sample("z", dist.Bernoulli(p))
            with pyro.plate("data", 3):
                pyro.sample("x", dist.Normal(z, 1.), obs=data)

    def guide(num_particles):
        pass

    elbo = TraceEnum_ELBO(max_plate_nesting=2)
    expected_logprob = -elbo.differentiable_loss(model, guide, num_particles=1)

    posterior_model = infer_discrete(config_enumerate(model, "parallel"),
                                     first_available_dim=-3)
    posterior_trace = poutine.trace(posterior_model).get_trace(num_particles=num_particles)
    actual_logprob = log_mean_prob(posterior_trace, particle_dim=-2)

    if nderivs == 0:
        assert_equal(expected_logprob, actual_logprob, prec=1e-3)
    elif nderivs == 1:
        expected_grad = grad(expected_logprob, [p])[0]
        actual_grad = grad(actual_logprob, [p])[0]
        assert_equal(expected_grad, actual_grad, prec=1e-3)
