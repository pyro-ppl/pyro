# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pytest
from pyro.infer.importance import psis_diagnostic
from pyro.infer.util import MultiFrameTensor
from tests.common import assert_equal


def xy_model():
    d = dist.Bernoulli(0.5)
    x_axis = pyro.plate('x_axis', 2, dim=-1)
    y_axis = pyro.plate('y_axis', 3, dim=-2)
    pyro.sample('b', d)
    with x_axis:
        pyro.sample('bx', d.expand_by([2]))
    with y_axis:
        pyro.sample('by', d.expand_by([3, 1]))
    with x_axis, y_axis:
        pyro.sample('bxy', d.expand_by([3, 2]))


def test_multi_frame_tensor():
    stacks = {}
    actual = MultiFrameTensor()
    tr = poutine.trace(xy_model).get_trace()
    for name, site in tr.nodes.items():
        if site["type"] == "sample":
            log_prob = site["fn"].log_prob(site["value"])
            stacks[name] = site["cond_indep_stack"]
            actual.add((site["cond_indep_stack"], log_prob))

    assert len(actual) == 4

    logp = math.log(0.5)
    expected = {
        'b': torch.ones(torch.Size()) * logp * (1 + 2 + 3 + 6),
        'bx': torch.ones(torch.Size((2,))) * logp * (1 + 1 + 3 + 3),
        'by': torch.ones(torch.Size((3, 1))) * logp * (1 + 2 + 1 + 2),
        'bxy': torch.ones(torch.Size((3, 2))) * logp * (1 + 1 + 1 + 1),
    }
    for name, expected_sum in expected.items():
        actual_sum = actual.sum_to(stacks[name])
        assert_equal(actual_sum, expected_sum, msg=name)


@pytest.mark.parametrize('max_particles', [250 * 1000, 500 * 1000])
@pytest.mark.parametrize('scale,krange', [(0.5, (0.7, 0.9)),
                                          (0.95, (0.05, 0.2))])
@pytest.mark.parametrize('zdim', [1, 5])
def test_psis_diagnostic(scale, krange, zdim, max_particles, num_particles=500 * 1000):

    def model(zdim=1, scale=1.0):
        with pyro.plate("x_axis", zdim, dim=-1):
            pyro.sample("z", dist.Normal(0.0, 1.0).expand([zdim]))

    def guide(zdim=1, scale=1.0):
        with pyro.plate("x_axis", zdim, dim=-1):
            pyro.sample("z", dist.Normal(0.0, scale).expand([zdim]))

    k = psis_diagnostic(model, guide, num_particles=num_particles, max_simultaneous_particles=max_particles,
                        zdim=zdim, scale=scale)
    assert k > krange[0] and k < krange[1]
