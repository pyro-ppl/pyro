import math

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.util import MultiFrameTensor
from tests.common import assert_equal


def xy_model():
    d = dist.Bernoulli(0.5)
    x_axis = pyro.iarange('x_axis', 2, dim=-1)
    y_axis = pyro.iarange('y_axis', 3, dim=-2)
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
