from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import torch

from pyro.distributions.util import broadcast_shape
from pyro.ops.sumproduct import logsumproductexp, memoized_sum_keepdim


def _check_tree_structure(tensor_tree):
    # Check for absence of diamonds, i.e. collider iaranges.
    # This is required to avoid loops in message passing.
    #
    # For example let f1, f2 be two CondIndepStackFrames, and consider
    # all nesting structures closed under intersection.
    # The following nesting structures are all valid:
    #
    #                 {f1}         {f2}           {}
    #                  |            |            /  \
    #  {f1,f2}      {f1,f2}      {f1,f2}      {f1}  {f2}
    #
    #           {}           {}           {}
    #            |            |            |
    #            |          {f1}          {f2}
    #            |            |            |
    #         {f1,f2}      {f1,f2}      {f1,f2}
    #
    # But the "diamond" nesting structure is invalid:
    #
    #         {}
    #        /  \
    #     {f1}  {f2}
    #        \  /
    #      {f1,f2}
    #
    # In this case the {f1,f2} context contains an enumerated variable that
    # depends on enumerated variables in both {f1} and {f2}.
    for t in tensor_tree:
        for u in tensor_tree:
            if not (u < t):
                continue
            for v in tensor_tree:
                if not (v < t):
                    continue
                if u <= v or v <= u:
                    continue
                left = ', '.join(sorted(f.name for f in u - v))
                right = ', '.join(sorted(f.name for f in v - u))
                raise ValueError("Expected tree-structured iarange nesting, but found "
                                 "dependencies on independent iarange sets [{}] and [{}]. "
                                 "Try converting one of the iaranges to an irange (but beware "
                                 "exponential cost in the size of that irange)"
                                 .format(left, right))


def _partition_terms(terms, dims):
    """
    Given a list of terms and a set of contraction dims, partitions the terms
    up into sets that must be contracted together. By separating these
    components we avoid broadcasting. This function should be deterministic.
    """
    assert all(dim < 0 for dim in dims)

    # Construct a bipartite graph between terms and the dims in which they
    # are enumerated. This conflates terms and dims (tensors and ints).
    neighbors = OrderedDict([(t, []) for t in terms] + [(d, []) for d in dims])
    for term in terms:
        for dim in range(-term.dim(), 0):
            if dim in dims and term.shape[dim] > 1:
                neighbors[term].append(dim)
                neighbors[dim].append(term)

    # Partition the bipartite graph into connected components for contraction.
    while neighbors:
        v, pending = neighbors.popitem()
        component = OrderedDict([(v, None)])  # used as an OrderedSet
        for v in pending:
            component[v] = None
        while pending:
            v = pending.pop()
            for v in neighbors.pop(v):
                if v not in component:
                    component[v] = None
                    pending.append(v)

        # Split this connected component into tensors and dims.
        component_terms = [v for v in component if isinstance(v, torch.Tensor)]
        component_dims = set(v for v in component if not isinstance(v, torch.Tensor))
        yield component_terms, component_dims


def _contract_component(tensor_tree, sum_dims, target_ordinal=None):
    """
    Contract out ``sum_dims`` in a tree of tensors in-place, via message
    passing. This reduces all tensors down to a single iarange context
    ``target_ordinal``, by default the minimum shared iarange context.

    This function should be deterministic.

    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param dict sum_dims: a dictionary mapping tensors to sets of dimensions
        (indexed from the right) that should be summed out.
    :param frozenset target_ordinal: An optional ordinal to which results will
        be contracted or broadcasted.
    """
    if target_ordinal is None:
        target_ordinal = frozenset.intersection(*tensor_tree)

    # First close the set of ordinals under intersection (greatest lower bound),
    # ensuring that the ordinals are arranged in a tree structure.
    tensor_tree.setdefault(target_ordinal, [])
    pending = list(tensor_tree)
    while pending:
        t = pending.pop()
        for u in list(tensor_tree):
            tu = t & u
            if tu not in tensor_tree:
                tensor_tree[tu] = []
                pending.append(tu)
    _check_tree_structure(tensor_tree)

    # Collect contraction dimension by ordinal.
    dims_tree = defaultdict(set)
    for t, terms in tensor_tree.items():
        dims_tree[t] = set.union(set(), *(sum_dims[term] for term in terms))

    # Recursively combine terms in different iarange contexts.
    while len(tensor_tree) > 1 or any(dims_tree.values()):
        leaf = max(tensor_tree, key=lambda t: len(t ^ target_ordinal))
        leaf_terms = tensor_tree.pop(leaf)
        leaf_dims = dims_tree.pop(leaf)
        remaining_dims = set.union(set(), *dims_tree.values())
        contract_dims = leaf_dims - remaining_dims
        contract_frames = frozenset()
        broadcast_frames = frozenset()
        if leaf - target_ordinal:
            # Contract out unneeded iaranges.
            contract_frames = frozenset.intersection(*(leaf - t for t in tensor_tree if t < leaf))
        elif target_ordinal - leaf:
            # Broadcast up needed iaranges.
            broadcast_frames = frozenset.intersection(*(t - leaf for t in tensor_tree if t > leaf))
        parent = leaf - contract_frames | broadcast_frames
        dims_tree[parent] |= leaf_dims & remaining_dims
        tensor_tree.setdefault(parent, [])

        for terms, dims in _partition_terms(leaf_terms, contract_dims):

            # Eliminate any enumeration dims via a logsumproductexp() contraction.
            if dims:
                shape = list(broadcast_shape(*set(x.shape for x in terms)))
                for dim in dims:
                    shape[dim] = 1
                shape.reverse()
                while shape and shape[-1] == 1:
                    shape.pop()
                shape.reverse()
                shape = tuple(shape)
                terms = [logsumproductexp(terms, shape)]

            for term in terms:
                # Eliminate extra iarange dims via .sum() contractions.
                for frame in sorted(contract_frames, key=lambda f: -f.dim):
                    term = memoized_sum_keepdim(term, frame.dim)

                tensor_tree[parent].append(term)


def contract_tensor_tree(tensor_tree, sum_dims):
    """
    Contract out ``sum_dims`` in a tree of tensors in-place, via
    message passing.

    This function should be deterministic and free of side effects.

    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param dict sum_dims: a dictionary mapping tensors to sets of dimensions
        (indexed from the right) that should be summed out.
    :returns: A contracted version of ``tensor_tree`
    :rtype: OrderedDict
    """
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, dict)

    ordinals = {term: t for t, terms in tensor_tree.items() for term in terms}
    all_terms = [term for terms in tensor_tree.values() for term in terms]
    all_dims = set.union(*sum_dims.values())
    contracted_tree = OrderedDict()

    # Split this tensor tree into connected components.
    for terms, dims in _partition_terms(all_terms, all_dims):
        component = OrderedDict()
        for term in terms:
            component.setdefault(ordinals[term], []).append(term)

        # Contract this connected component down to a single tensor.
        _contract_component(component, sum_dims)
        assert len(component) == 1
        assert len(next(iter(component.values()))) == 1
        for t, terms in component.items():
            contracted_tree.setdefault(t, []).extend(terms)

    return contracted_tree


def contract_to_tensor(tensor_tree, sum_dims, target_ordinal):
    """
    Contract out ``sum_dims`` in a tree of tensors, via message
    passing. This reduces all terms down to a single tensor in the iarange
    context specified by ``target_ordinal``.

    This function should be deterministic and free of side effects.

    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param dict sum_dims: a dictionary mapping tensors to sets of dimensions
        (indexed from the right) that should be summed out.
    :param frozendset target_ordinal: An optional ordinal to which results will
        be contracted or broadcasted.
    :returns: a single tensor
    :rtype: torch.Tensor
    """
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, dict)
    assert isinstance(target_ordinal, frozenset)

    contracted_tree = OrderedDict((t, terms[:]) for t, terms in tensor_tree.items())
    _contract_component(contracted_tree, sum_dims, target_ordinal)
    t, terms = contracted_tree.popitem()
    assert t == target_ordinal
    term = sum(terms)

    # Broadcast to any missing iaranges via .expand().
    shape = list(term.shape)
    for frame in target_ordinal:
        shape = [1] * (-frame.dim - len(shape)) + shape
        shape[frame.dim] = frame.size
    shape = torch.Size(shape)
    if term.shape != shape:
        term = term.expand(shape)

    return term
