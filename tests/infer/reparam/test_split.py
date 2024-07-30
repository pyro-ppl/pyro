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

from .util import check_init_reparam

event_shape_splits_dim = pytest.mark.parametrize(
    "event_shape,splits,dim",
    [
        ((6,), [2, 1, 3], -1),
        (
            (
                2,
                5,
            ),
            [2, 3],
            -1,
        ),
        ((4, 2), [1, 3], -2),
        ((2, 3, 1), [1, 2], -2),
    ],
    ids=str,
)


batch_shape = pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)


@event_shape_splits_dim
@batch_shape
def test_normal(batch_shape, event_shape, splits, dim):
    shape = batch_shape + event_shape
    loc = torch.empty(shape).uniform_(-1.0, 1.0).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()

    def model():
        with pyro.plate_stack("plates", batch_shape):
            pyro.sample("x", dist.Normal(loc, scale).to_event(len(event_shape)))

    # Run without reparam.
    trace = poutine.trace(model).get_trace()
    expected_value = trace.nodes["x"]["value"]
    expected_log_prob = trace.log_prob_sum()
    expected_grads = grad(expected_log_prob, [loc, scale], create_graph=True)

    # Run with reparam.
    split_values = {
        "x_split_{}".format(i): xi
        for i, xi in enumerate(expected_value.split(splits, dim))
    }
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


@event_shape_splits_dim
@batch_shape
def test_init(batch_shape, event_shape, splits, dim):
    shape = batch_shape + event_shape
    loc = torch.empty(shape).uniform_(-1.0, 1.0)
    scale = torch.empty(shape).uniform_(0.5, 1.5)

    def model():
        with pyro.plate_stack("plates", batch_shape):
            return pyro.sample("x", dist.Normal(loc, scale).to_event(len(event_shape)))

    check_init_reparam(model, SplitReparam(splits, dim))


@batch_shape
def test_transformed_distribution(batch_shape):
    num_samples = 10

    transform = dist.transforms.StackTransform(
        [
            dist.transforms.OrderedTransform(),
            dist.transforms.DiscreteCosineTransform(),
            dist.transforms.HaarTransform(),
        ],
        dim=-1,
    )

    num_transforms = len(transform.transforms)

    def model():
        scale_tril = pyro.sample("scale_tril", dist.LKJCholesky(num_transforms, 1))
        with pyro.plate_stack("plates", batch_shape):
            x_dist = dist.TransformedDistribution(
                dist.MultivariateNormal(
                    torch.zeros(num_samples, num_transforms), scale_tril=scale_tril
                ).to_event(1),
                [transform],
            )
            return pyro.sample("x", x_dist)

    assert model().shape == batch_shape + (num_samples, num_transforms)

    pyro.clear_param_store()
    guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    guide_sites = guide()

    assert guide_sites["x"].shape == batch_shape + (num_samples, num_transforms)

    for sections in [[1, 1, 1], [1, 2], [2, 1]]:
        split_model = pyro.poutine.reparam(
            model, config={"x": SplitReparam(sections, -1)}
        )

        pyro.clear_param_store()
        guide = pyro.infer.autoguide.AutoMultivariateNormal(split_model)
        guide_sites = guide()

        for n, section in enumerate(sections):
            assert guide_sites[f"x_split_{n}"].shape == batch_shape + (
                num_samples,
                section,
            )
