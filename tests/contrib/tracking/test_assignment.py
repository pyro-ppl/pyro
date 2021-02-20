# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

import logging

import pyro
import pyro.distributions as dist
from pyro.contrib.tracking.assignment import MarginalAssignment, MarginalAssignmentPersistent, MarginalAssignmentSparse
from tests.common import assert_equal

INF = float('inf')
logger = logging.getLogger(__name__)


def assert_finite(tensor, name):
    assert ((tensor - tensor) == 0).all(), 'bad {}: {}'.format(tensor, name)


def logit(p):
    return p.log() - (-p).log1p()


def dense_to_sparse(assign_logits):
    num_detections, num_objects = assign_logits.shape
    edges = assign_logits.new_tensor([[j, i] for j in range(num_detections) for i in range(num_objects)],
                                     dtype=torch.long).t()
    assign_logits = assign_logits[edges[0], edges[1]]
    return edges, assign_logits


def sparse_to_dense(num_objects, num_detections, edges, assign_logits):
    result = assign_logits.new_empty(num_detections, num_objects).fill_(-INF)
    result[edges[0], edges[1]] = assign_logits
    return result


def test_dense_smoke():
    num_objects = 4
    num_detections = 2
    pyro.set_rng_seed(0)
    exists_logits = torch.zeros(num_objects)
    assign_logits = logit(torch.tensor([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.5],
    ]))
    assert assign_logits.shape == (num_detections, num_objects)

    solver = MarginalAssignment(exists_logits, assign_logits, bp_iters=5)

    assert solver.exists_dist.batch_shape == (num_objects,)
    assert solver.exists_dist.event_shape == ()
    assert solver.assign_dist.batch_shape == (num_detections,)
    assert solver.assign_dist.event_shape == ()
    assert solver.assign_dist.probs.shape[-1] == num_objects + 1  # true + spurious

    # test dense matches sparse
    edges, assign_logits = dense_to_sparse(assign_logits)
    other = MarginalAssignmentSparse(num_objects, num_detections, edges, exists_logits, assign_logits, bp_iters=5)
    assert_equal(other.exists_dist.probs, solver.exists_dist.probs, prec=1e-3)
    assert_equal(other.assign_dist.probs, solver.assign_dist.probs, prec=1e-3)


def test_sparse_smoke():
    num_objects = 4
    num_detections = 2
    pyro.set_rng_seed(0)
    exists_logits = torch.zeros(num_objects)
    edges = exists_logits.new_tensor([
        [0, 0, 1, 0, 1, 0],
        [0, 1, 1, 2, 2, 3],
    ], dtype=torch.long)
    assign_logits = logit(torch.tensor([0.99, 0.8, 0.2, 0.2, 0.8, 0.9]))
    assert assign_logits.shape == edges.shape[1:]

    solver = MarginalAssignmentSparse(num_objects, num_detections, edges,
                                      exists_logits, assign_logits, bp_iters=5)

    assert solver.exists_dist.batch_shape == (num_objects,)
    assert solver.exists_dist.event_shape == ()
    assert solver.assign_dist.batch_shape == (num_detections,)
    assert solver.assign_dist.event_shape == ()
    assert solver.assign_dist.probs.shape[-1] == num_objects + 1  # true + spurious

    # test dense matches sparse
    assign_logits = sparse_to_dense(num_objects, num_detections, edges, assign_logits)
    other = MarginalAssignment(exists_logits, assign_logits, bp_iters=5)
    assert_equal(other.exists_dist.probs, solver.exists_dist.probs, prec=1e-3)
    assert_equal(other.assign_dist.probs, solver.assign_dist.probs, prec=1e-3)


def test_sparse_grid_smoke():

    def my_existence_prior(ox, oy):
        return -0.5

    def my_assign_prior(ox, oy, dx, dy):
        return 0.0

    num_detections = 3 * 3
    detections = [[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]]
    num_objects = 2 * 2
    objects = [[0, 1],
               [2, 3]]
    edges = []
    edge_coords = []
    for x in range(2):
        for y in range(2):
            object_id = objects[x][y]
            for dx in [0, 1]:
                for dy in [0, 1]:
                    detection_id = detections[x + dx][y + dy]
                    edges.append((detection_id, object_id))
                    edge_coords.append((x, y, x + dx, y + dy))

    exists_logits = torch.empty(num_objects)
    edges = exists_logits.new_tensor(edges, dtype=torch.long).t()
    assert edges.shape == (2, 4 * 4)

    for x in range(2):
        for y in range(2):
            object_id = objects[x][y]
            exists_logits[object_id] = my_existence_prior(x, y)
    assign_logits = exists_logits.new_tensor([my_assign_prior(ox, oy, dx, dy)
                                              for ox, oy, dx, dy in edge_coords])
    assign = MarginalAssignmentSparse(num_objects, num_detections, edges,
                                      exists_logits, assign_logits, bp_iters=10)
    assert isinstance(assign.assign_dist, dist.Categorical)


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
        assert_finite(e_grad, 'dassign_probs/dexists_logits')
        assert_finite(a_grad, 'dassign_probs/dassign_logits')


@pytest.mark.parametrize('e', [-1., 0., 1.])
@pytest.mark.parametrize('a', [-1., 0., 1.])
def test_flat_exact_1_1(e, a):
    exists_logits = torch.tensor([e])
    assign_logits = torch.tensor([[a]])
    expected = MarginalAssignment(exists_logits, assign_logits, None)
    actual = MarginalAssignment(exists_logits, assign_logits, 10)
    assert_equal(expected.exists_dist.probs, actual.exists_dist.probs)
    assert_equal(expected.assign_dist.probs, actual.assign_dist.probs)


@pytest.mark.parametrize('e', [-1., 0., 1.])
@pytest.mark.parametrize('a11', [-1., 0., 1.])
@pytest.mark.parametrize('a21', [-1., 0., 1.])
def test_flat_exact_2_1(e, a11, a21):
    exists_logits = torch.tensor([e])
    assign_logits = torch.tensor([[a11], [a21]])
    expected = MarginalAssignment(exists_logits, assign_logits, None)
    actual = MarginalAssignment(exists_logits, assign_logits, 10)
    assert_equal(expected.exists_dist.probs, actual.exists_dist.probs)
    assert_equal(expected.assign_dist.probs, actual.assign_dist.probs)


@pytest.mark.parametrize('e1', [-1., 0., 1.])
@pytest.mark.parametrize('e2', [-1., 0., 1.])
@pytest.mark.parametrize('a11', [-1., 0., 1.])
@pytest.mark.parametrize('a12', [-1., 0., 1.])
def test_flat_exact_1_2(e1, e2, a11, a12):
    exists_logits = torch.tensor([e1, e2])
    assign_logits = torch.tensor([[a11, a12]])
    expected = MarginalAssignment(exists_logits, assign_logits, None)
    actual = MarginalAssignment(exists_logits, assign_logits, 10)
    assert_equal(expected.exists_dist.probs, actual.exists_dist.probs)
    assert_equal(expected.assign_dist.probs, actual.assign_dist.probs)


@pytest.mark.parametrize('e1', [-1., 1.])
@pytest.mark.parametrize('e2', [-1., 1.])
@pytest.mark.parametrize('a11', [-1., 1.])
@pytest.mark.parametrize('a12', [-1., 1.])
@pytest.mark.parametrize('a22', [-1., 1.])
def test_flat_exact_2_2(e1, e2, a11, a12, a22):
    a21 = -INF
    exists_logits = torch.tensor([e1, e2])
    assign_logits = torch.tensor([[a11, a12], [a21, a22]])
    expected = MarginalAssignment(exists_logits, assign_logits, None)
    actual = MarginalAssignment(exists_logits, assign_logits, 10)
    assert_equal(expected.exists_dist.probs, actual.exists_dist.probs)
    assert_equal(expected.assign_dist.probs, actual.assign_dist.probs)


@pytest.mark.parametrize('num_detections', [1, 2, 3, 4])
@pytest.mark.parametrize('num_objects', [1, 2, 3, 4])
def test_flat_bp_vs_exact(num_objects, num_detections):
    exists_logits = -2 * torch.rand(num_objects)
    assign_logits = -2 * torch.rand(num_detections, num_objects)
    expected = MarginalAssignment(exists_logits, assign_logits, None)
    actual = MarginalAssignment(exists_logits, assign_logits, 10)
    # these should only approximately agree
    assert_equal(expected.exists_dist.probs, actual.exists_dist.probs, prec=0.01)
    assert_equal(expected.assign_dist.probs, actual.assign_dist.probs, prec=0.01)


@pytest.mark.parametrize('num_frames', [1, 2, 3, 4])
@pytest.mark.parametrize('num_objects', [1, 2, 3, 4])
@pytest.mark.parametrize('bp_iters', [None, 30], ids=['enum', 'bp'])
def test_flat_vs_persistent(num_objects, num_frames, bp_iters):
    exists_logits = -2 * torch.rand(num_objects)
    assign_logits = -2 * torch.rand(num_frames, num_objects)
    flat = MarginalAssignment(exists_logits, assign_logits, bp_iters)
    full = MarginalAssignmentPersistent(exists_logits, assign_logits.unsqueeze(1), bp_iters)
    assert_equal(flat.exists_dist.probs, full.exists_dist.probs)
    assert_equal(flat.assign_dist.probs, full.assign_dist.probs.squeeze(1))


@pytest.mark.parametrize('num_detections', [1, 2, 3, 4])
@pytest.mark.parametrize('num_frames', [1, 2, 3, 4])
@pytest.mark.parametrize('num_objects', [1, 2, 3, 4])
def test_persistent_bp_vs_exact(num_objects, num_frames, num_detections):
    exists_logits = -2 * torch.rand(num_objects)
    assign_logits = 2 * torch.rand(num_frames, num_detections, num_objects) - 1
    expected = MarginalAssignmentPersistent(exists_logits, assign_logits, None)
    actual = MarginalAssignmentPersistent(exists_logits, assign_logits, 30)
    # these should only approximately agree
    assert_equal(expected.exists_dist.probs, actual.exists_dist.probs, prec=0.05)
    assert_equal(expected.assign_dist.probs, actual.assign_dist.probs, prec=0.05)


@pytest.mark.parametrize('e1', [-1., 1.])
@pytest.mark.parametrize('e2', [-1., 1.])
@pytest.mark.parametrize('e3', [-1., 1.])
@pytest.mark.parametrize('bp_iters, bp_momentum', [(3, 0.), (30, 0.5)], ids=['momentum', 'none'])
def test_persistent_exact_5_4_3(e1, e2, e3, bp_iters, bp_momentum):
    exists_logits = torch.tensor([e1, e2, e3])
    assign_logits = 2 * torch.rand(5, 4, 3) - 1
    # this has tree-shaped connectivity and should lead to exact inference
    mask = torch.tensor([[[1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
                         [[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]],
                         [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
                         [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]],
                         [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]], dtype=torch.bool)
    assign_logits[~mask] = -INF
    expected = MarginalAssignmentPersistent(exists_logits, assign_logits, None)
    actual = MarginalAssignmentPersistent(exists_logits, assign_logits, bp_iters, bp_momentum)
    assert_equal(expected.exists_dist.probs, actual.exists_dist.probs)
    assert_equal(expected.assign_dist.probs, actual.assign_dist.probs)
    logger.debug(actual.exists_dist.probs)
    logger.debug(actual.assign_dist.probs)


@pytest.mark.parametrize('num_detections', [1, 2, 3])
@pytest.mark.parametrize('num_frames', [1, 2, 3])
@pytest.mark.parametrize('num_objects', [1, 2])
@pytest.mark.parametrize('bp_iters', [None, 30], ids=['enum', 'bp'])
def test_persistent_independent_subproblems(num_objects, num_frames, num_detections, bp_iters):
    # solve a random assignment problem
    exists_logits_1 = -2 * torch.rand(num_objects)
    assign_logits_1 = 2 * torch.rand(num_frames, num_detections, num_objects) - 1
    assignment_1 = MarginalAssignmentPersistent(exists_logits_1, assign_logits_1, bp_iters)
    exists_probs_1 = assignment_1.exists_dist.probs
    assign_probs_1 = assignment_1.assign_dist.probs

    # solve another random assignment problem
    exists_logits_2 = -2 * torch.rand(num_objects)
    assign_logits_2 = 2 * torch.rand(num_frames, num_detections, num_objects) - 1
    assignment_2 = MarginalAssignmentPersistent(exists_logits_2, assign_logits_2, bp_iters)
    exists_probs_2 = assignment_2.exists_dist.probs
    assign_probs_2 = assignment_2.assign_dist.probs

    # solve a unioned assignment problem
    exists_logits = torch.cat([exists_logits_1, exists_logits_2])
    assign_logits = torch.full((num_frames, num_detections * 2, num_objects * 2), -INF)
    assign_logits[:, :num_detections, :num_objects] = assign_logits_1
    assign_logits[:, num_detections:, num_objects:] = assign_logits_2
    assignment = MarginalAssignmentPersistent(exists_logits, assign_logits, bp_iters)
    exists_probs = assignment.exists_dist.probs
    assign_probs = assignment.assign_dist.probs

    # check agreement
    assert_equal(exists_probs_1, exists_probs[:num_objects])
    assert_equal(exists_probs_2, exists_probs[num_objects:])
    assert_equal(assign_probs_1[:, :, :-1], assign_probs[:, :num_detections, :num_objects])
    assert_equal(assign_probs_1[:, :, -1], assign_probs[:, :num_detections, -1])
    assert_equal(assign_probs_2[:, :, :-1], assign_probs[:, num_detections:, num_objects:-1])
    assert_equal(assign_probs_2[:, :, -1], assign_probs[:, num_detections:, -1])
