from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import grad

from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from tests.common import xfail_if_not_implemented

INF = float('inf')


def assert_finite(tensor, name):
    assert ((tensor - tensor) == 0).all(), 'bad {}: {}'.format(tensor, name)


@pytest.mark.parametrize('bp_iters', [None, 10], ids=['enum', 'bp'])
def test_persistent_smoke(bp_iters):
    exists_logits = torch.tensor([-1., -1., -2.], requires_grad=True)
    assign_logits = torch.tensor([[[-1., -INF, -INF],
                                   [-2., -2., -INF]],
                                  [[-1., -2., -3.],
                                   [-2., -2., -1.]],
                                  [[-1., -2., -3.],
                                   [-2., -2., -1.]],
                                  [[-1., -1., 1.],
                                   [1., 1., -1.]]], requires_grad=True)

    with xfail_if_not_implemented():
        assignment = MarginalAssignmentPersistent(exists_logits, assign_logits, bp_iters=bp_iters)
    assert assignment.num_frames == 4
    assert assignment.num_detections == 2
    assert assignment.num_objects == 3

    assign_dist = assignment.assign_dist
    exists_dist = assignment.exists_dist
    assert_finite(exists_dist.probs, 'exists_probs')
    assert_finite(assign_dist.probs, 'assign_probs')

    for exists in exists_dist.enumerate_support():
        log_prob = exists_dist.log_prob(exists).sum()
        e_grad, a_grad = grad(log_prob, [exists_logits, assign_logits], create_graph=True)
        assert_finite(e_grad, 'dexists_probs/dexists_logits')
        assert_finite(a_grad, 'dexists_probs/dassign_logits')

    for assign in assign_dist.enumerate_support():
        log_prob = assign_dist.log_prob(assign).sum()
        e_grad, a_grad = grad(log_prob, [exists_logits, assign_logits], create_graph=True)
        assert_finite(e_grad, 'dexists_probs/dexists_logits')
        assert_finite(a_grad, 'dexists_probs/dassign_logits')
