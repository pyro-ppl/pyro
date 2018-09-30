from __future__ import absolute_import, division, print_function

import heapq
import itertools
import operator
from collections import defaultdict

import numpy as np
from six.moves import reduce


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


def footprint(sizes, dims):
    factors = map(sizes.__getitem__, dims)
    return reduce(operator.mul, factors, 1)


def get_candidate(output, sizes, footprints, dim_to_keys, k1, k2):
    both = k1 & k2
    either = k1 | k2
    k12 = frozenset(dim for dim in either
                    if dim in output or len(dim_to_keys[dim]) > 1 + int(dim in both))
    cost = footprint(sizes, k12) - footprints[k1] - footprints[k2]
    return cost, k1, k2, k12


def push_candidate(output, sizes, footprints, dim_to_keys, k1, k2s, queue):
    if not k2s:
        return
    candidate = min(get_candidate(output, sizes, footprints, dim_to_keys, k1, k2)
                    for k2 in k2s)
    heapq.heappush(queue, candidate)


def ssa_optimize(inputs, output, sizes):
    """
    This has an interface similar to :func:`optimize` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    output = frozenset(output)
    ssa_path = []

    # Deduplicate by eagerly computing Hadamard products.
    remaining = {}  # key -> ssa_id
    ssa_ids = itertools.count(len(inputs))
    for ssa_id, key in enumerate(map(frozenset, inputs)):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            ssa_id = next(ssa_ids)
        remaining[key] = ssa_id

    # Initialize footprints of each tensor and contraction candidates.
    footprints = {key: footprint(sizes, key) for key in remaining}
    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key:
            dim_to_keys[dim].add(key)

    # Find initial candidate contractions.
    queue = []
    for dim, keys in dim_to_keys.items():
        keys = list(keys)
        for i, k1 in enumerate(keys):
            k2s = keys[:i]
            push_candidate(output, sizes, footprints, dim_to_keys, k1, k2s, queue)

    # Greedily contract pairs of tensors.
    while queue:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue  # candidate is obsolete

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1:
            dim_to_keys[dim].remove(k1)
        for dim in k2:
            dim_to_keys[dim].remove(k2)
        if k12 in remaining:
            ssa_id12 = remaining[k12]
            ssa_path.append((ssa_id1, ssa_id2, ssa_id12))
        else:
            ssa_path.append((ssa_id1, ssa_id2))
            footprints[k12] = footprint(sizes, k12)
            for dim in k12:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)

        # Find new candidate contractions.
        k1 = k12
        k2s = set(k2 for dim in k1 for k2 in dim_to_keys[dim] if k2 != k1)
        push_candidate(output, sizes, footprints, dim_to_keys, k1, k2s, queue)

    # Compute remaining outer products.
    ssa_path.append(tuple(remaining.values()))
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
    ssa_path = ssa_optimize(inputs, output, sizes)
    return ssa_to_linear(ssa_path)


__all__ = ['optimize']
