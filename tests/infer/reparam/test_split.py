# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import SplitReparam
from tests.common import assert_close


@pytest.mark.parametrize("event_shape,splits,dim", [
    ((6,), [2, 1, 3], -1),
    ((2, 5,), [2, 3], -1),
    ((4, 2), [1, 3], -2),
    ((2, 3, 1), [1, 2], -2),
], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_normal(batch_shape, event_shape, splits, dim):
    shape = batch_shape + event_shape
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()

    def model():
        with pyro.plate_stack("plates", batch_shape):
            pyro.sample("x",
                        dist.Normal(loc, scale)
                            .to_event(len(event_shape)))

    # Run without reparam.
    trace = poutine.trace(model).get_trace()
    expected_value = trace.nodes["x"]["value"]
    expected_log_prob = trace.log_prob_sum()
    expected_grads = grad(expected_log_prob, [loc, scale], create_graph=True)

    # Run with reparam.
    split_values = {
        "x_split_{}".format(i): xi
        for i, xi in enumerate(expected_value.split(splits, dim))}
    rep = SplitReparam(splits, dim)
    reparam_model = poutine.reparam(model, {"x": rep})
    reparam_model = poutine.condition(reparam_model, split_values)
    trace = poutine.trace(reparam_model).get_trace()
    assert all(name in trace.nodes for name in split_values)
    assert isinstance(trace.nodes["x"]["fn"], dist.Delta)
    assert trace.nodes["x"]["fn"].batch_shape == batch_shape
    assert trace.nodes["x"]["fn"].event_shape == event_shape

    # Check values.
    actual_value = trace.nodes["x"]["value"]
    assert_close(actual_value, expected_value, atol=0.1)

    # Check log prob.
    actual_log_prob = trace.log_prob_sum()
    assert_close(actual_log_prob, expected_log_prob)
    actual_grads = grad(actual_log_prob, [loc, scale], create_graph=True)
    assert_close(actual_grads, expected_grads)
