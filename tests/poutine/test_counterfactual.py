# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from tests.common import assert_equal, assert_not_equal


def _item(x):
    if isinstance(x, torch.Tensor):
        x = x.item()
    return x


@pytest.mark.parametrize('intervene,observe,flip', [
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, True, True),
])
def test_counterfactual_query(intervene, observe, flip):
    # x -> y -> z -> w

    sites = ["x", "y", "z", "w"]
    observations = {"x": 1., "y": None, "z": 1., "w": 1.}
    interventions = {"x": None, "y": 0., "z": 2., "w": 1.}

    def model():
        x = _item(pyro.sample("x", dist.Normal(0, 1)))
        y = _item(pyro.sample("y", dist.Normal(x, 1)))
        z = _item(pyro.sample("z", dist.Normal(y, 1)))
        w = _item(pyro.sample("w", dist.Normal(z, 1)))
        return dict(x=x, y=y, z=z, w=w)

    if not flip:
        if intervene:
            model = poutine.do(model, data=interventions)
        if observe:
            model = poutine.condition(model, data=observations)
    elif flip and intervene and observe:
        model = poutine.do(
            poutine.condition(model, data=observations),
            data=interventions)

    tr = poutine.trace(model).get_trace()
    actual_values = tr.nodes["_RETURN"]["value"]
    for name in sites:
        # case 1: purely observational query like poutine.condition
        if not intervene and observe:
            if observations[name] is not None:
                assert tr.nodes[name]['is_observed']
                assert_equal(observations[name], actual_values[name])
                assert_equal(observations[name], tr.nodes[name]['value'])
            if interventions[name] != observations[name]:
                assert_not_equal(interventions[name], actual_values[name])
        # case 2: purely interventional query like old poutine.do
        elif intervene and not observe:
            assert not tr.nodes[name]['is_observed']
            if interventions[name] is not None:
                assert_equal(interventions[name], actual_values[name])
            assert_not_equal(observations[name], tr.nodes[name]['value'])
            assert_not_equal(interventions[name], tr.nodes[name]['value'])
        # case 3: counterfactual query mixing intervention and observation
        elif intervene and observe:
            if observations[name] is not None:
                assert tr.nodes[name]['is_observed']
                assert_equal(observations[name], tr.nodes[name]['value'])
            if interventions[name] is not None:
                assert_equal(interventions[name], actual_values[name])
            if interventions[name] != observations[name]:
                assert_not_equal(interventions[name], tr.nodes[name]['value'])
