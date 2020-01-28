# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import numbers

import torch

import pyro.distributions as dist
from pyro.util import warn_if_nan


def _product(factors):
    result = 1.
    for factor in factors:
        result = result * factor
    return result


def _exp(value):
    if isinstance(value, numbers.Number):
        return math.exp(value)
    return value.exp()


class MarginalAssignment:
    """
    Computes marginal data associations between objects and detections.

    This assumes that each detection corresponds to zero or one object,
    and each object corresponds to zero or more detections. Specifically
    this does not assume detections have been partitioned into frames of
    mutual exclusion as is common in 2-D assignment problems.

    :param torch.Tensor exists_logits: a tensor of shape ``[num_objects]``
        representing per-object factors for existence of each potential object.
    :param torch.Tensor assign_logits: a tensor of shape
        ``[num_detections, num_objects]`` representing per-edge factors of
        assignment probability, where each edge denotes that a given detection
        associates with a single object.
    :param int bp_iters: optional number of belief propagation iterations. If
        unspecified or ``None`` an expensive exact algorithm will be used.

    :ivar int num_detections: the number of detections
    :ivar int num_objects: the number of (potentially existing) objects
    :ivar pyro.distributions.Bernoulli exists_dist: a mean field posterior
        distribution over object existence.
    :ivar pyro.distributions.Categorical assign_dist: a mean field posterior
        distribution over the object (or None) to which each detection
        associates.  This has ``.event_shape == (num_objects + 1,)`` where the
        final element denotes spurious detection, and
        ``.batch_shape == (num_frames, num_detections)``.
    """
    def __init__(self, exists_logits, assign_logits, bp_iters=None):
        assert exists_logits.dim() == 1, exists_logits.shape
        assert assign_logits.dim() == 2, assign_logits.shape
        assert assign_logits.shape[-1] == exists_logits.shape[-1]
        self.num_detections, self.num_objects = assign_logits.shape

        # Clamp to avoid NANs.
        exists_logits = exists_logits.clamp(min=-40, max=40)
        assign_logits = assign_logits.clamp(min=-40, max=40)

        # This does all the work.
        if bp_iters is None:
            exists, assign = compute_marginals(exists_logits, assign_logits)
        else:
            exists, assign = compute_marginals_bp(exists_logits, assign_logits, bp_iters)

        # Wrap the results in Distribution objects.
        # This adds a final logit=0 element denoting spurious detection.
        padded_assign = torch.nn.functional.pad(assign, (0, 1), "constant", 0.0)
        self.assign_dist = dist.Categorical(logits=padded_assign)
        self.exists_dist = dist.Bernoulli(logits=exists)


class MarginalAssignmentSparse:
    """
    A cheap sparse version of :class:`MarginalAssignment`.

    :param int num_detections: the number of detections
    :param int num_objects: the number of (potentially existing) objects
    :param torch.LongTensor edges: a ``[2, num_edges]``-shaped tensor of
        (detection, object) index pairs specifying feasible associations.
    :param torch.Tensor exists_logits: a tensor of shape ``[num_objects]``
        representing per-object factors for existence of each potential object.
    :param torch.Tensor assign_logits: a tensor of shape ``[num_edges]``
        representing per-edge factors of assignment probability, where each
        edge denotes that a given detection associates with a single object.
    :param int bp_iters: optional number of belief propagation iterations. If
        unspecified or ``None`` an expensive exact algorithm will be used.

    :ivar int num_detections: the number of detections
    :ivar int num_objects: the number of (potentially existing) objects
    :ivar pyro.distributions.Bernoulli exists_dist: a mean field posterior
        distribution over object existence.
    :ivar pyro.distributions.Categorical assign_dist: a mean field posterior
        distribution over the object (or None) to which each detection
        associates.  This has ``.event_shape == (num_objects + 1,)`` where the
        final element denotes spurious detection, and
        ``.batch_shape == (num_frames, num_detections)``.
    """
    def __init__(self, num_objects, num_detections, edges, exists_logits, assign_logits, bp_iters):
        assert edges.dim() == 2, edges.shape
        assert edges.shape[0] == 2, edges.shape
        assert exists_logits.shape == (num_objects,), exists_logits.shape
        assert assign_logits.shape == edges.shape[1:], assign_logits.shape
        self.num_objects = num_objects
        self.num_detections = num_detections
        self.edges = edges

        # Clamp to avoid NANs.
        exists_logits = exists_logits.clamp(min=-40, max=40)
        assign_logits = assign_logits.clamp(min=-40, max=40)

        # This does all the work.
        exists, assign = compute_marginals_sparse_bp(
            num_objects, num_detections, edges, exists_logits, assign_logits, bp_iters)

        # Wrap the results in Distribution objects.
        # This adds a final logit=0 element denoting spurious detection.
        padded_assign = torch.full((num_detections, num_objects + 1), -float('inf'),
                                   dtype=assign.dtype, device=assign.device)
        padded_assign[:, -1] = 0
        padded_assign[edges[0], edges[1]] = assign
        self.assign_dist = dist.Categorical(logits=padded_assign)
        self.exists_dist = dist.Bernoulli(logits=exists)


class MarginalAssignmentPersistent:
    """
    This computes marginal distributions of a multi-frame multi-object
    data association problem with an unknown number of persistent objects.

    The inputs are factors in a factor graph (existence probabilites for each
    potential object and assignment probabilities for each object-detection
    pair), and the outputs are marginal distributions of posterior existence
    probability of each potential object and posterior assignment probabilites
    of each object-detection pair.

    This assumes a shared (maximum) number of detections per frame; to handle
    variable number of detections, simply set corresponding elements of
    ``assign_logits`` to ``-float('inf')``.

    :param torch.Tensor exists_logits: a tensor of shape ``[num_objects]``
        representing per-object factors for existence of each potential object.
    :param torch.Tensor assign_logits: a tensor of shape
        ``[num_frames, num_detections, num_objects]`` representing per-edge
        factors of assignment probability, where each edge denotes that at a
        given time frame a given detection associates with a single object.
    :param int bp_iters: optional number of belief propagation iterations. If
        unspecified or ``None`` an expensive exact algorithm will be used.
    :param float bp_momentum: optional momentum to use for belief propagation.
        Should be in the interval ``[0,1)``.

    :ivar int num_frames: the number of time frames
    :ivar int num_detections: the (maximum) number of detections per frame
    :ivar int num_objects: the number of (potentially existing) objects
    :ivar pyro.distributions.Bernoulli exists_dist: a mean field posterior
        distribution over object existence.
    :ivar pyro.distributions.Categorical assign_dist: a mean field posterior
        distribution over the object (or None) to which each detection
        associates.  This has ``.event_shape == (num_objects + 1,)`` where the
        final element denotes spurious detection, and
        ``.batch_shape == (num_frames, num_detections)``.
    """
    def __init__(self, exists_logits, assign_logits, bp_iters=None, bp_momentum=0.5):
        assert exists_logits.dim() == 1, exists_logits.shape
        assert assign_logits.dim() == 3, assign_logits.shape
        assert assign_logits.shape[-1] == exists_logits.shape[-1]
        self.num_frames, self.num_detections, self.num_objects = assign_logits.shape

        # Clamp to avoid NANs.
        exists_logits = exists_logits.clamp(min=-40, max=40)
        assign_logits = assign_logits.clamp(min=-40, max=40)

        # This does all the work.
        if bp_iters is None:
            exists, assign = compute_marginals_persistent(exists_logits, assign_logits)
        else:
            exists, assign = compute_marginals_persistent_bp(
                exists_logits, assign_logits, bp_iters, bp_momentum)

        # Wrap the results in Distribution objects.
        # This adds a final logit=0 element denoting spurious detection.
        padded_assign = torch.nn.functional.pad(assign, (0, 1), "constant", 0.0)
        self.assign_dist = dist.Categorical(logits=padded_assign)
        self.exists_dist = dist.Bernoulli(logits=exists)
        assert self.assign_dist.batch_shape == (self.num_frames, self.num_detections)
        assert self.exists_dist.batch_shape == (self.num_objects,)


def compute_marginals(exists_logits, assign_logits):
    """
    This implements exact inference of pairwise marginals via
    enumeration. This is very expensive and is only useful for testing.

    See :class:`MarginalAssignment` for args and problem description.
    """
    num_detections, num_objects = assign_logits.shape
    assert exists_logits.shape == (num_objects,)
    dtype = exists_logits.dtype
    device = exists_logits.device

    exists_probs = torch.zeros(2, num_objects, dtype=dtype, device=device)  # [not exist, exist]
    assign_probs = torch.zeros(num_detections, num_objects + 1, dtype=dtype, device=device)
    for assign in itertools.product(range(num_objects + 1), repeat=num_detections):
        assign_part = sum(assign_logits[j, i] for j, i in enumerate(assign) if i < num_objects)
        for exists in itertools.product(*[[1] if i in assign else [0, 1] for i in range(num_objects)]):
            exists_part = sum(exists_logits[i] for i, e in enumerate(exists) if e)
            prob = _exp(exists_part + assign_part)
            for i, e in enumerate(exists):
                exists_probs[e, i] += prob
            for j, i in enumerate(assign):
                assign_probs[j, i] += prob

    # Convert from probs to logits.
    exists = exists_probs.log()
    assign = assign_probs.log()
    exists = exists[1] - exists[0]
    assign = assign[:, :-1] - assign[:, -1:]
    warn_if_nan(exists, 'exists')
    warn_if_nan(assign, 'assign')
    return exists, assign


def compute_marginals_bp(exists_logits, assign_logits, bp_iters):
    """
    This implements approximate inference of pairwise marginals via
    loopy belief propagation, adapting the approach of [1].

    See :class:`MarginalAssignment` for args and problem description.

    [1] Jason L. Williams, Roslyn A. Lau (2014)
        Approximate evaluation of marginal association probabilities with
        belief propagation
        https://arxiv.org/abs/1209.6299
    """
    message_e_to_a = torch.zeros_like(assign_logits)
    message_a_to_e = torch.zeros_like(assign_logits)
    for i in range(bp_iters):
        message_e_to_a = -(message_a_to_e - message_a_to_e.sum(0, True) - exists_logits).exp().log1p()
        joint = (assign_logits + message_e_to_a).exp()
        message_a_to_e = (assign_logits - torch.log1p(joint.sum(1, True) - joint)).exp().log1p()
        warn_if_nan(message_e_to_a, 'message_e_to_a iter {}'.format(i))
        warn_if_nan(message_a_to_e, 'message_a_to_e iter {}'.format(i))

    # Convert from probs to logits.
    exists = exists_logits + message_a_to_e.sum(0)
    assign = assign_logits + message_e_to_a
    warn_if_nan(exists, 'exists')
    warn_if_nan(assign, 'assign')
    return exists, assign


def compute_marginals_sparse_bp(num_objects, num_detections, edges,
                                exists_logits, assign_logits, bp_iters):
    """
    This implements approximate inference of pairwise marginals via
    loopy belief propagation, adapting the approach of [1].

    See :class:`MarginalAssignmentSparse` for args and problem description.

    [1] Jason L. Williams, Roslyn A. Lau (2014)
        Approximate evaluation of marginal association probabilities with
        belief propagation
        https://arxiv.org/abs/1209.6299
    """
    exists_factor = exists_logits[edges[1]]

    def sparse_sum(x, dim, keepdim=False):
        assert dim in (0, 1)
        x = (torch.zeros([num_objects, num_detections][dim], dtype=x.dtype, device=x.device)
                  .scatter_add_(0, edges[1 - dim], x))
        if keepdim:
            x = x[edges[1 - dim]]
        return x

    message_e_to_a = torch.zeros_like(assign_logits)
    message_a_to_e = torch.zeros_like(assign_logits)
    for i in range(bp_iters):
        message_e_to_a = -(message_a_to_e - sparse_sum(message_a_to_e, 0, True) - exists_factor).exp().log1p()
        joint = (assign_logits + message_e_to_a).exp()
        message_a_to_e = (assign_logits - torch.log1p(sparse_sum(joint, 1, True) - joint)).exp().log1p()
        warn_if_nan(message_e_to_a, 'message_e_to_a iter {}'.format(i))
        warn_if_nan(message_a_to_e, 'message_a_to_e iter {}'.format(i))

    # Convert from probs to logits.
    exists = exists_logits + sparse_sum(message_a_to_e, 0)
    assign = assign_logits + message_e_to_a
    warn_if_nan(exists, 'exists')
    warn_if_nan(assign, 'assign')
    return exists, assign


def compute_marginals_persistent(exists_logits, assign_logits):
    """
    This implements exact inference of pairwise marginals via
    enumeration. This is very expensive and is only useful for testing.

    See :class:`MarginalAssignmentPersistent` for args and problem description.
    """
    num_frames, num_detections, num_objects = assign_logits.shape
    assert exists_logits.shape == (num_objects,)
    dtype = exists_logits.dtype
    device = exists_logits.device

    total = 0
    exists_probs = torch.zeros(num_objects, dtype=dtype, device=device)
    assign_probs = torch.zeros(num_frames, num_detections, num_objects, dtype=dtype, device=device)
    for exists in itertools.product([0, 1], repeat=num_objects):
        exists = [i for i, e in enumerate(exists) if e]
        exists_part = _exp(sum(exists_logits[i] for i in exists))

        # The remaining variables are conditionally independent conditioned on exists.
        assign_parts = []
        assign_sums = []
        for t in range(num_frames):
            assign_map = {}
            for n in range(1 + min(len(exists), num_detections)):
                for objects in itertools.combinations(exists, n):
                    for detections in itertools.permutations(range(num_detections), n):
                        assign = tuple(zip(objects, detections))
                        assign_map[assign] = _exp(sum(assign_logits[t, j, i] for i, j in assign))
            assign_parts.append(assign_map)
            assign_sums.append(sum(assign_map.values()))

        prob = exists_part * _product(assign_sums)
        total += prob
        for i in exists:
            exists_probs[i] += prob
        for t in range(num_frames):
            other_part = exists_part * _product(assign_sums[:t] + assign_sums[t + 1:])
            for assign, assign_part in assign_parts[t].items():
                prob = other_part * assign_part
                for i, j in assign:
                    assign_probs[t, j, i] += prob

    # Convert from probs to logits.
    exists = exists_probs.log() - (total - exists_probs).log()
    assign = assign_probs.log() - (total - assign_probs.sum(-1, True)).log()
    warn_if_nan(exists, 'exists')
    warn_if_nan(assign, 'assign')
    return exists, assign


def compute_marginals_persistent_bp(exists_logits, assign_logits, bp_iters, bp_momentum=0.5):
    """
    This implements approximate inference of pairwise marginals via
    loopy belief propagation, adapting the approach of [1], [2].

    See :class:`MarginalAssignmentPersistent` for args and problem description.

    [1] Jason L. Williams, Roslyn A. Lau (2014)
        Approximate evaluation of marginal association probabilities with
        belief propagation
        https://arxiv.org/abs/1209.6299
    [2] Ryan Turner, Steven Bottone, Bhargav Avasarala (2014)
        A Complete Variational Tracker
        https://papers.nips.cc/paper/5572-a-complete-variational-tracker.pdf
    """
    # This implements forward-backward message passing among three sets of variables:
    #
    #   a[t,j] ~ Categorical(num_objects + 1), detection -> object assignment
    #   b[t,i] ~ Categorical(num_detections + 1), object -> detection assignment
    #     e[i] ~ Bernonulli, whether each object exists
    #
    # Only assign = a and exists = e are returned.
    assert 0 <= bp_momentum < 1, bp_momentum
    old, new = bp_momentum, 1 - bp_momentum
    num_frames, num_detections, num_objects = assign_logits.shape
    dtype = assign_logits.dtype
    device = assign_logits.device
    message_b_to_a = torch.zeros(num_frames, num_detections, num_objects, dtype=dtype, device=device)
    message_a_to_b = torch.zeros(num_frames, num_detections, num_objects, dtype=dtype, device=device)
    message_b_to_e = torch.zeros(num_frames, num_objects, dtype=dtype, device=device)
    message_e_to_b = torch.zeros(num_frames, num_objects, dtype=dtype, device=device)

    for i in range(bp_iters):
        odds_a = (assign_logits + message_b_to_a).exp()
        message_a_to_b = (old * message_a_to_b +
                          new * (assign_logits - (odds_a.sum(2, True) - odds_a).log1p()))
        message_b_to_e = (old * message_b_to_e +
                          new * message_a_to_b.exp().sum(1).log1p())
        message_e_to_b = (old * message_e_to_b +
                          new * (exists_logits + message_b_to_e.sum(0) - message_b_to_e))
        odds_b = message_a_to_b.exp()
        message_b_to_a = (old * message_b_to_a -
                          new * ((-message_e_to_b).exp().unsqueeze(1) + (1 + odds_b.sum(1, True) - odds_b)).log())

        warn_if_nan(message_a_to_b, 'message_a_to_b iter {}'.format(i))
        warn_if_nan(message_b_to_e, 'message_b_to_e iter {}'.format(i))
        warn_if_nan(message_e_to_b, 'message_e_to_b iter {}'.format(i))
        warn_if_nan(message_b_to_a, 'message_b_to_a iter {}'.format(i))

    # Convert from probs to logits.
    exists = exists_logits + message_b_to_e.sum(0)
    assign = assign_logits + message_b_to_a
    warn_if_nan(exists, 'exists')
    warn_if_nan(assign, 'assign')
    return exists, assign
