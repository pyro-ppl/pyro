import math

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.util import MultiFrameTensor, MultiViewTensor
from tests.common import assert_equal


def test_multiview_tensor_add():
    x = MultiViewTensor()
    x.add(torch.ones(2, 3, 4))
    assert_equal(x[torch.Size([2, 3, 4])], torch.ones(2, 3, 4))
    x.add(x)
    x.add(torch.ones(2, 3, 5))
    assert_equal(x[torch.Size([2, 3, 4])], 2.0 * torch.ones(2, 3, 4))
    x.add(x)
    assert_equal(x[torch.Size([2, 3, 4])], 4.0 * torch.ones(2, 3, 4))
    assert_equal(x[torch.Size([2, 3, 5])], 2.0 * torch.ones(2, 3, 5))
    y = MultiViewTensor(1.5 * torch.ones(2, 3, 4))
    x.add(y)
    assert_equal(x[torch.Size([2, 3, 4])], 5.5 * torch.ones(2, 3, 4))


def test_multiview_tensor_sum_leftmost_all_but():
    x = MultiViewTensor(1.5 * torch.ones(2, 3, 4))
    x.add(2.5 * torch.ones(1, 3, 4))
    x.add(3.5 * torch.ones(2, 3, 1))
    result = x.sum_leftmost_all_but(2)
    assert_equal(result[torch.Size([3, 4])], 5.5 * torch.ones(3, 4))
    assert_equal(result[torch.Size([3, 1])], 7.0 * torch.ones(3, 1))


def test_multiview_tensor_contract_as():
    x = MultiViewTensor(1.5 * torch.ones(2, 1, 4))
    x.add(2.5 * torch.ones(1, 3, 4))
    x.add(3.5 * torch.ones(2, 3, 1))
    x.add(4.5 * torch.ones(3, 1))
    result214 = x.contract_as(torch.ones(2, 1, 4))
    result134 = x.contract_as(torch.ones(1, 3, 4))
    result231 = x.contract_as(torch.ones(2, 3, 1))
    assert_equal(result214, (1.5 + 7.5 + 10.5 + 13.5) * torch.ones(2, 1, 4))
    assert_equal(result134, (2.5 + 3.0 + 7.0 + 4.5) * torch.ones(1, 3, 4))
    assert_equal(result231, (3.5 + 10.0 + 6.0 + 4.5) * torch.ones(2, 3, 1))
    assert x.contract_as(torch.ones(3, 1)).shape == (3, 1)


def xy_model():
    d = dist.Bernoulli(0.5)
    x_axis = pyro.iarange('x_axis', 2, dim=-1)
    y_axis = pyro.iarange('y_axis', 3, dim=-2)
    pyro.sample('b', d)
    with x_axis:
        pyro.sample('bx', d.reshape([2]))
    with y_axis:
        pyro.sample('by', d.reshape([3, 1]))
    with x_axis, y_axis:
        pyro.sample('bxy', d.reshape([3, 2]))


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
