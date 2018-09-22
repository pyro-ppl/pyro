from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import torch
from opt_einsum import shared_intermediates
from six import add_metaclass

from pyro.distributions.util import broadcast_shape
from pyro.ops.einsum import contract
from pyro.ops.sumproduct import logsumproductexp


@add_metaclass(ABCMeta)
class TensorRing(object):
    @abstractmethod
    def dims(self, term):
        raise NotImplementedError

    @abstractmethod
    def sumproduct(self, terms, dims):
        raise NotImplementedError

    @abstractmethod
    def product(self, term, ordinal):
        raise NotImplementedError

    @abstractmethod
    def broadcast(self, tensor, ordinal):
        raise NotImplementedError


class UnpackedLogRing(TensorRing):
    def __init__(self, cache=None):
        if cache is None:
            cache = {}
        self._cache = cache

    def dims(self, term):
        for dim in range(-term.dim(), 0):
            if term.shape[dim] > 1:
                yield dim

    def sumproduct(self, terms, dims):
        if not dims:
            return sum(terms)
        shape = list(broadcast_shape(*set(x.shape for x in terms)))
        for dim in dims:
            shape[dim] = 1
        shape.reverse()
        while shape and shape[-1] == 1:
            shape.pop()
        shape.reverse()
        shape = tuple(shape)
        return logsumproductexp(terms, shape)

    def product(self, term, ordinal):
        for frame in sorted(ordinal, key=lambda f: -f.dim):
            if term.shape[frame.dim] != 1:
                key = 'product', id(term), frame.dim
                if key in self._cache:
                    term = self._cache[key]
                else:
                    term = term.sum(frame.dim, keepdim=True)
                    self._cache[key] = term
        return term

    def broadcast(self, term, ordinal):
        shape = list(term.shape)
        for frame in ordinal:
            shape = [1] * (-frame.dim - len(shape)) + shape
            shape[frame.dim] = frame.size
        shape = torch.Size(shape)
        if term.shape == shape:
            return term
        key = 'expand', id(term), shape
        if key in self._cache:
            return self._cache[key]
        term = term.expand(shape)
        self._cache[key] = term
        return term


class PackedLogRing(TensorRing):
    def __init__(self, inputs, operands, cache=None):
        if cache is None:
            cache = {}
        self._batch_size = {}
        for dims, term in zip(inputs, operands):
            cache['tensor', id(term)] = term
            cache['dims', id(term)] = dims
            for dim, size in zip(dims, term.shape):
                self._batch_size[dim] = size
        self._cache = cache

    def dims(self, term):
        return self._cache['dims', id(term)]

    def sumproduct(self, terms, dims):
        inputs = [self.dims(term) for term in terms]
        output = ''.join(sorted(set(sum(inputs)) - set(dims)))
        equation = ''.join(inputs) + '->' + output
        term = contract(equation, *terms, backend='pyro.ops.einsum.torch_log')
        self._cache['tensor', id(term)] = term
        self._cache['dims', id(term)] = output
        return term

    def product(self, term, ordinal):
        dims = self.dims(term)
        for dim in sorted(ordinal, reverse=True):
            pos = dims.find(dim)
            if pos != -1:
                key = 'product', id(term), dim
                if key in self._cache:
                    term = self._cache[key]
                else:
                    term = term.sum(pos)
                    dims = dims.replace(dim, '')
                    self._cache[key] = term
                    self._cache['dim', id(term)] = dims
        return term

    def broadcast(self, term, ordinal):
        dims = self.dims(term)
        missing_dims = ''.join(sorted(set(ordinal) - set(dims)))
        if missing_dims:
            key = 'broadcast', id(term), missing_dims
            if key in self._cache:
                term = self._cache[key]
            else:
                missing_shape = tuple(self._batch_size[dim] for dim in missing_dims)
                term = term.expand(missing_shape + term.shape)
                dims = missing_dims + dims
                self._cache[key] = term
                self._cache['dims', id(term)] = dims
        return term


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
                left = ', '.join(sorted(getattr(f, 'name', str(f)) for f in u - v))
                right = ', '.join(sorted(getattr(f, 'name', str(f)) for f in v - u))
                raise ValueError("Expected tree-structured iarange nesting, but found "
                                 "dependencies on independent iarange sets [{}] and [{}]. "
                                 "Try converting one of the iaranges to an irange (but beware "
                                 "exponential cost in the size of that irange)"
                                 .format(left, right))


def _partition_terms(ring, terms, dims):
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
        for dim in ring.dims(term):
            if dim in dims:
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


def _contract_component(ring, tensor_tree, sum_dims, target_ordinal=None):
    """
    Contract out ``sum_dims`` in a tree of tensors in-place, via message
    passing. This reduces all tensors down to a single iarange context
    ``target_ordinal``, by default the minimum shared iarange context.

    This function should be deterministic.
    This function has side-effects: it modifies ``tensor_tree`` in-place.

    :param TensorRing ring: an algebraic ring defining tensor operations.
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

        # Split the current node into connected components.
        for terms, dims in _partition_terms(ring, leaf_terms, contract_dims):

            # Eliminate any enumeration dims via a sumproduct contraction.
            if dims:
                terms = [ring.sumproduct(terms, dims)]

            for term in terms:
                if contract_frames:
                    # Eliminate extra iarange dims via product contractions.
                    term = ring.product(term, contract_frames)
                elif broadcast_frames:
                    # Broadcast to any missing iaranges via .expand().
                    term = ring.broadcast(term, broadcast_frames)
                tensor_tree[parent].append(term)


def contract_tensor_tree(tensor_tree, sum_dims, ring=None, cache=None):
    """
    Contract out ``sum_dims`` in a tree of tensors via message passing.

    This function should be deterministic and free of side effects.

    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param dict sum_dims: a dictionary mapping tensors to sets of dimensions
        (indexed from the right) that should be summed out.
    :param TensorRing ring: an algebraic ring defining tensor operations.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :returns: A contracted version of ``tensor_tree``
    :rtype: OrderedDict
    """
    if ring is None:
        ring = UnpackedLogRing(cache=cache)
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, dict)
    assert isinstance(ring, TensorRing)

    ordinals = {term: t for t, terms in tensor_tree.items() for term in terms}
    all_terms = [term for terms in tensor_tree.values() for term in terms]
    all_dims = set.union(*sum_dims.values())
    contracted_tree = OrderedDict()

    # Split this tensor tree into connected components.
    for terms, dims in _partition_terms(ring, all_terms, all_dims):
        component = OrderedDict()
        for term in terms:
            component.setdefault(ordinals[term], []).append(term)

        # Contract this connected component down to a single tensor.
        _contract_component(ring, component, sum_dims)
        assert len(component) == 1
        assert len(next(iter(component.values()))) == 1
        for t, terms in component.items():
            contracted_tree.setdefault(t, []).extend(terms)

    return contracted_tree


def contract_to_tensor(tensor_tree, sum_dims, target_ordinal, ring=None, cache=None):
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
    :param TensorRing ring: an algebraic ring defining tensor operations.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :returns: a single tensor
    :rtype: torch.Tensor
    """
    if ring is None:
        ring = UnpackedLogRing(cache=cache)
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, dict)
    assert isinstance(target_ordinal, frozenset)
    assert isinstance(ring, TensorRing)

    contracted_tree = OrderedDict((t, terms[:]) for t, terms in tensor_tree.items())
    _contract_component(ring, contracted_tree, sum_dims, target_ordinal)
    t, terms = contracted_tree.popitem()
    assert t == target_ordinal
    return ring.sumproduct(terms, {})


def ubersum(equation, *operands, **kwargs):
    """
    Generalized batched einsum via tensor message passing.

    :param str equation: an einsum equation, optionally with multuple outputs.
    :param torch.Tensor operands: a collection of tensors
    :param str batch_dims: a string of batch dims.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :return: a tuple of tensors of requested shape.
    :rtype: tuple
    """
    # Extract kwargs.
    cache = kwargs.pop('cache', None)
    batch_dims = kwargs.pop('batch_dims', '')
    backend = kwargs.pop('backend', 'pyro.ops.einsum.torch_log')
    if backend != 'pyro.ops.einsum.torch_log':
        raise NotImplementedError

    # Parse generalized einsum equation.
    if '.' in equation:
        raise NotImplementedError('ubsersum does not yet support ellipsis notation')
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    outputs = outputs.split(',')
    assert len(inputs) == len(outputs)
    assert all(isinstance(x, torch.Tensor) for x in operands)

    # Construct a tensor tree shared by all outputs.
    tensor_tree = OrderedDict()
    max_ordinal = frozenset(batch_dims)
    for dims, term in zip(inputs, operands):
        assert len(dims) == term.dim()
        ordinal = frozenset(dims) & max_ordinal
        tensor_tree.setdefault(ordinal, []).append(term)

    # Compute outputs, sharing intermediate computations.
    results = []
    with shared_intermediates(cache) as cache:
        ring = PackedLogRing(inputs, operands, cache=cache)
        for output in outputs:
            nosum_dims = set(batch_dims + output)
            sum_dims = {term: set(dims) - nosum_dims for dims, term in zip(inputs, operands)}
            target_ordinal = frozenset(output) & max_ordinal
            term = contract_to_tensor(tensor_tree, sum_dims, target_ordinal, ring=ring)
            dims = ring.dims(term)
            if dims != output:
                term = term.permute(*map(dims.index, output))
            results.append(term)
    return tuple(results)
