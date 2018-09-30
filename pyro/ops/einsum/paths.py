from __future__ import absolute_import, division, print_function

import heapq
import itertools
from collections import defaultdict

import numpy as np


def ssa_to_linear(ssa_path):
    """
    Convert a path with static single assignment ids to a path with recycled
    linear ids.
    """
    ids = np.arange(sum(map(len, ssa_path)), dtype=np.int64)
    path = []
    for i, ssa_ids in enumerate(ssa_path):
        path.append(tuple(ids[ssa_id] for ssa_id in ssa_ids))
        for ssa_id in ssa_ids:
            ids[ssa_id:] -= 1
    return path


def linear_to_ssa(path):
    """
    Convert a path with recycled linear ids to a path with static single
    assignment ids.
    """
    num_inputs = sum(map(len, path)) - len(path) + 1
    linear_to_ssa = list(range(num_inputs))
    new_ids = itertools.count(num_inputs)
    ssa_path = []
    for ids in path:
        ssa_path.append(tuple(linear_to_ssa[id_] for id_ in ids))
        for id_ in sorted(ids, reverse=True):
            del linear_to_ssa[id_]
        linear_to_ssa.append(next(new_ids))
    return ssa_path


def _footprint(sizes, dims):
    result = 1
    for dim in dims:
        result *= sizes[dim]
    return result


def _get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    cost = _footprint(sizes, k12) - footprints[k1] - footprints[k2]
    id1 = remaining[k1]
    id2 = remaining[k2]
    cost = cost, min(id1, id2), max(id1, id2)  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue):
    if not k2s:
        return
    candidate = min(_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2)
                    for k2 in k2s)
    heapq.heappush(queue, candidate)


def _update_ref_counts(dim_to_keys, dim_ref_counts, dims):
    for dim in dims:
        count = len(dim_to_keys[dim])
        if count <= 1:
            dim_ref_counts[2].discard(dim)
            dim_ref_counts[3].discard(dim)
        elif count == 2:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].discard(dim)
        else:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].add(dim)


def _ssa_optimize(inputs, output, sizes):
    """
    This has an interface similar to :func:`optimize` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    # A dim common to all terms might as well be an output dim.
    inputs = list(map(frozenset, inputs))
    output = frozenset(output) | frozenset.intersection(*inputs)

    # Deduplicate shapes by eagerly computing Hadamard products.
    remaining = {}  # key -> ssa_id
    ssa_ids = itertools.count(len(inputs))
    ssa_path = []
    for ssa_id, key in enumerate(inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id

    # Compute footprints of each tensor.
    footprints = {key: _footprint(sizes, key) for key in remaining}

    # Keep track of possible contraction dims.
    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)

    # Keep track of the number of tensors using each dim; when the dim is no longer
    # used it can be contracted. Since we specialize to binary ops, we only care about
    # ref counts of >=2 or >=3.
    dim_ref_counts = {
        count: set(dim for dim, keys in dim_to_keys.items() if len(keys) >= count) - output
        for count in [2, 3]}

    # Find initial candidate contractions.
    queue = []
    for dim, keys in dim_to_keys.items():
        keys = list(keys)
        for i, k1 in enumerate(keys):
            k2s = keys[:i]
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue)

    # Greedily contract pairs of tensors.
    while queue:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue  # candidate is obsolete

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id2, ssa_id1))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            footprints[k12] = _footprint(sizes, k12)
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)

        # Find new candidate contractions.
        k1 = k12
        k2s = set(k2 for dim in k1 for k2 in dim_to_keys[dim] if k2 != k1)
        _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue)

    # Greedily compute pairwise outer products.
    queue = [(len(key & output), ssa_id, key) for key, ssa_id in remaining.items()]
    heapq.heapify(queue)
    _, ssa_id1, k1 = heapq.heappop(queue)
    while queue:
        _, ssa_id2, k2 = heapq.heappop(queue)
        ssa_path.append((ssa_id1, ssa_id2))
        k12 = (k1 | k2) & output
        cost = len(k12)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(queue, (cost, ssa_id12, k12))

    # Perform one final reduction to match output shape.
    ssa_path.append((ssa_id1,))
    return ssa_path


def optimize(inputs, output, sizes):
    """
    Produces an optimization path similar to the greedy strategy
    :func:`opt_einsum.paths.greedy`. This optimizer is cheaper and less
    accurate than the default ``opt_einsum`` optimizer.

    :param list inputs: A list of input shapes. These can be strings or sets or
        frozensets of characters.
    :param str output: An output shape. This can be a string or set or
        frozenset of characters.
    :param dict sizes: A mapping from dimensions (characters in inputs) to ints
        that are the sizes of those dimensions.
    :return: An optimization path: a list if tuples of contraction indices.
    rtype: list
    """
    ssa_path = _ssa_optimize(inputs, output, sizes)
    return ssa_to_linear(ssa_path)


__all__ = ['optimize']
