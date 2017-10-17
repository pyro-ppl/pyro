import torch
from torch.autograd import Variable
import pytest

import pyro
import pyro.distributions as dist
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.util import zero_grads
from tests.common import assert_equal

# A simple Gaussian mixture model with tiny data.
gmm_data = Variable(torch.Tensor([1, 2]))


def gmm_model():
    p = pyro.param("p", Variable(torch.Tensor([0.2]), requires_grad=True))
    sigma = pyro.param("sigma", Variable(torch.Tensor([1.0]), requires_grad=True))
    mus = pyro.param("mus", Variable(torch.Tensor([1, 2]), requires_grad=True))
    for i in pyro.irange("data", len(gmm_data)):
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.size() == (1, 1)
        z = z.long().data[0, 0]
        pyro.observe("x_{}".format(i), dist.DiagNormal(mus[z], sigma), gmm_data[i])


def gmm_guide():
    for i in pyro.irange("data", len(gmm_data)):
        p = pyro.param("p_{}".format(i), Variable(torch.Tensor([0.2]), requires_grad=True))
        z = pyro.sample("z_{}".format(i), dist.Bernoulli(p))
        assert z.size() == (1, 1)


@pytest.mark.parametrize('num_particles', [1, 2])
@pytest.mark.parametrize('enum_discrete', [False, True], ids=['sample', 'sum'])
def test_trace_elbo_traces(num_particles, enum_discrete):
    pyro.clear_param_store()
    elbo = Trace_ELBO(num_particles=num_particles, enum_discrete=enum_discrete)
    traces = list(elbo._iter_traces(gmm_model, gmm_guide))
    if enum_discrete:
        assert len(traces) == num_particles * 2 ** len(gmm_data)
    else:
        assert len(traces) == num_particles


@pytest.mark.parametrize('num_particles', [1, 2])
@pytest.mark.parametrize('enum_discrete', [False, True], ids=['sample', 'sum'])
def test_trace_elbo_loss(num_particles, enum_discrete):
    pyro.clear_param_store()
    elbo = Trace_ELBO(num_particles=num_particles, enum_discrete=enum_discrete)
    traces = list(elbo._iter_traces(gmm_model, gmm_guide))

    loss_v1 = elbo._compute_loss(traces)
    loss_v2 = elbo._compute_loss_and_grads(traces)
    assert_equal(loss_v1, loss_v2)


# TODO Fix these tests.
@pytest.mark.parametrize('enum_discrete', [False, True], ids=['sample', 'sum'])
def test_trace_elbo_grads(enum_discrete):
    pyro.clear_param_store()
    elbo = Trace_ELBO(num_particles=200, enum_discrete=enum_discrete)
    traces = list(elbo._iter_traces(gmm_model, gmm_guide))
    params = ["p", "sigma", "mus"] + ["p_{}".format(i) for i in range(len(gmm_data))]

    # Compute gradients via loss.
    loss = elbo._compute_loss(traces)
    loss.backward(retain_graph=True)
    expected_grads = {name: pyro.param(name).grad.data.clone() for name in params}

    # Compute gradients via surrogate loss.
    zero_grads(pyro.param(name) for name in params)
    elbo._compute_loss_and_grads(traces)
    actual_grads = {name: pyro.param(name).grad.data.clone() for name in params}

    for name in params:
        print('{} {}{}{}'.format(name, '-' * 30, actual_grads[name], expected_grads[name]))
        assert_equal(actual_grads[name], expected_grads[name], prec=0.1,
                     msg='{} gradients differ:{}{}'.format(name, actual_grads[name],
                                                           expected_grads[name]))
