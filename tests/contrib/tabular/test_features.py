from __future__ import absolute_import, division, print_function

import logging

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.contrib.tabular import Boolean, Discrete, Real
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.util import torch_isnan
from tests.common import assert_close, xfail_if_not_implemented


class Discrete5(Discrete):
    def __init__(self, name):
        super(Discrete5, self).__init__(name, 5)


@pytest.mark.parametrize('size', [1, 2, 100])
@pytest.mark.parametrize('MyFeature', [Boolean, Discrete5, Real])
def test_smoke(MyFeature, size):

    @poutine.broadcast
    def model(data=None, size=None):
        if data is not None:
            size = len(data)
        num_components = 10
        weights = pyro.param("component_weights",
                             torch.ones(num_components) / num_components,
                             constraint=constraints.simplex)
        membership_dist = dist.Categorical(weights)

        f = MyFeature("foo")
        shared = f.sample_shared()
        with pyro.plate("components", num_components):
            group = f.sample_group(shared)
        with pyro.plate("data", size):
            component = pyro.sample("component", membership_dist,
                                    infer={"enumerate": "parallel"})
            return pyro.sample("obs", f.value_dist(group, component), obs=data)

    # Generate synthetic data.
    pyro.set_rng_seed(0)
    data = model(size=size)
    assert len(data) == size

    # Train the model on data.
    pyro.set_rng_seed(1)
    pyro.clear_param_store()
    guide = AutoDelta(poutine.block(model, hide=["component"]))
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    optim = Adam({'lr': 0.1})
    svi = SVI(model, guide, optim, elbo)
    losses = []
    MyFeature("foo").init(data)
    for step in range(10):
        loss = svi.step(data)
        losses.append(loss)
        assert not torch_isnan(loss)
        logging.info('step {} loss = {}'.format(step, loss))
    if size > 1:
        assert losses[-1] < losses[0]


@pytest.mark.parametrize('partitions', [(), (5, 10), (1, 2, 3, 18, 19), (10, 10)])
@pytest.mark.parametrize('MyFeature', [Boolean, Discrete5, Real])
def test_summary(MyFeature, partitions):
    # Sample shared and group parameters.
    num_components = 7
    feature = MyFeature("foo")
    shared = feature.sample_shared()
    with pyro.plate("components", num_components):
        group = feature.sample_group(shared)

    # Sample data from the prior.
    num_rows = 20
    component = dist.Categorical(torch.ones(num_components)).sample((num_rows,))
    data = feature.value_dist(group, component).sample()
    assert len(data) == num_rows

    # Evaluate likelihood of collated data.
    log_probs = feature.value_dist(group, component).log_prob(data)
    expected_log_prob = torch.zeros(num_components).scatter_add_(0, component, log_probs)
    assert expected_log_prob.shape == (num_components,)

    # Create a pseudo dataset with matching statistics.
    with xfail_if_not_implemented():
        summary = feature.summary(group)
    for begin, end in zip((0,) + partitions, partitions + (num_rows,)):
        summary.scatter_update(component[begin: end], data[begin: end])
    pseudo_scale, pseudo_data = summary.as_scaled_data()
    pseudo_component = torch.arange(num_components).unsqueeze(-1)
    log_probs = feature.value_dist(group, pseudo_component).log_prob(pseudo_data)
    actual_log_prob = (log_probs * pseudo_scale).sum(-1)
    assert actual_log_prob.shape == (num_components,)
    assert_close(actual_log_prob, expected_log_prob)
