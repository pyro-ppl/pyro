from __future__ import absolute_import, division, print_function

import itertools
import math
import numbers
import warnings

import torch

import pyro.distributions as dist


def _warn_if_nan(tensor, name):
    if torch.isnan(tensor).any():
        warnings.warn('Encountered nan elements in {}'.format(name))
    if tensor.requires_grad:
        tensor.register_hook(lambda x: _warn_if_nan(x, name))


def _product(factors):
    result = 1
    for factor in factors:
        result *= factor
    return result


def _exp(value):
    if isinstance(value, numbers.Number):
        return math.exp(value)
    return value.exp()


class MarginalAssignmentPersistent(object):
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

    :param torch.Tensor exists_logits: a tensor of shape `[num_objects]`
        representing per-object factors for existence of each potential object.
    :param torch.Tensor assign_logits: a tensor of shape
        `[num_frames, num_detections, num_objects]` representing per-edge
        factors of assignment probability, where each edge denotes that at a
        given time frame a given detection associates with a single object.
    :param int bp_iters: optional number of belief propagation iterations. If
        unspecified or ``None`` an expensive exact algorithm will be used.
    :ivar int num_frames: the number of time frames
    :ivar int num_detections: the (maximum) number of detections per frame
    :ivar int num_objects: the number of (potentially existing) objects
    :ivar pyro.distributions.Bernoulli exists_dist: a mean field distribution
        over object existence.
    :ivar pyro.distributions.Categorical assign_dist: a mean field distribution
        over the object (or None) to which each detection associates.
        This has ``.event_shape == (num_objects + 1,)`` where the final element
        denotes spurious detection, and
        ``.batch_shape == (num_frames, num_detections)``.
    """
    def __init__(self, exists_logits, assign_logits, bp_iters=None):
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
            exists, assign = compute_marginals_persistent_bp(exists_logits, assign_logits, bp_iters)

        # Wrap the results in Distribution objects.
        # This adds a final logit=0 element denoting spurious detection.
        padded_assign = torch.nn.functional.pad(assign, (0, 1), "constant", 0.0)
        self.assign_dist = dist.Categorical(logits=padded_assign)
        self.exists_dist = dist.Bernoulli(logits=exists)
        assert self.assign_dist.batch_shape == (self.num_frames, self.num_detections)
        assert self.exists_dist.batch_shape == (self.num_objects,)


def compute_marginals_persistent(exists_logits, assign_logits):
    """
    This implements exact inference of pairwise marginals via
    enumeration. This is very expensive and is only useful for testing.
    """
    num_frames, num_detections, num_objects = assign_logits.shape
    assert exists_logits.shape == (num_objects,)

    total = 0
    exists_probs = exists_logits.new_zeros(num_objects)
    assign_probs = assign_logits.new_zeros(num_frames, num_detections, num_objects)
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

    # convert from probs to logits
    exists = exists_probs.log() - (total - exists_probs).log()
    assign = assign_probs.log() - (total - assign_probs.sum(-1, True)).log()
    _warn_if_nan(exists, 'exists')
    _warn_if_nan(assign, 'assign')
    return exists, assign


def compute_marginals_persistent_bp(exists_logits, assign_logits, bp_iters):
    """
    This implements approximate inference of pairwise marginals via
    loopy belief propagation, adapting the approach of [1], [2].

    [1] Jason L. Williams, Roslyn A. Lau (2014)
        Approximate evaluation of marginal association probabilities with
        belief propagation
        https://arxiv.org/abs/1209.6299
    [2] Ryan Turner, Steven Bottone, Bhargav Avasarala (2014)
        A Complete Variational Tracker
        https://papers.nips.cc/paper/5572-a-complete-variational-tracker.pdf
    """
    raise NotImplementedError
