# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import warnings
from collections import OrderedDict, defaultdict

import opt_einsum
import torch
from opt_einsum import shared_intermediates

from pyro.ops.rings import BACKEND_TO_RING, LogRing
from pyro.util import ignore_jit_warnings


def _check_plates_are_sensible(output_dims, nonoutput_ordinal):
    if output_dims and nonoutput_ordinal:
        raise ValueError(u"It is nonsensical to preserve a plated dim without preserving "
                         u"all of that dim's plates, but found '{}' without '{}'"
                         .format(output_dims, ','.join(nonoutput_ordinal)))


def _check_tree_structure(parent, leaf):
    if parent == leaf:
        raise NotImplementedError(
            "Expected tree-structured plate nesting, but found "
            "dependencies on independent plates [{}]. "
            "Try converting one of the vectorized plates to a sequential plate (but beware "
            "exponential cost in the size of the sequence)"
            .format(', '.join(getattr(f, 'name', str(f)) for f in leaf)))


def _partition_terms(ring, terms, dims):
    """
    Given a list of terms and a set of contraction dims, partitions the terms
    up into sets that must be contracted together. By separating these
    components we avoid broadcasting.

    This function should be deterministic and free of side effects.
    """
    # Construct a bipartite graph between terms and the dims in which they
    # are enumerated. This conflates terms and dims (tensors and ints).
    neighbors = OrderedDict([(t, []) for t in terms] + [(d, []) for d in sorted(dims)])
    for term in terms:
        for dim in term._pyro_dims:
            if dim in dims:
                neighbors[term].append(dim)
                neighbors[dim].append(term)

    # Partition the bipartite graph into connected components for contraction.
    components = []
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
        if component_terms:
            component_dims = set(v for v in component if not isinstance(v, torch.Tensor))
            components.append((component_terms, component_dims))
    return components


def _contract_component(ring, tensor_tree, sum_dims, target_dims):
    """
    Contract out ``sum_dims - target_dims`` in a tree of tensors in-place, via
    message passing. This reduces all tensors down to a single tensor in the
    minimum plate context.

    This function should be deterministic.
    This function has side-effects: it modifies ``tensor_tree``.

    :param pyro.ops.rings.Ring ring: an algebraic ring defining tensor
        operations.
    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param set sum_dims: the complete set of sum-contractions dimensions
        (indexed from the right). This is needed to distinguish sum-contraction
        dimensions from product-contraction dimensions.
    :param set target_dims: An subset of ``sum_dims`` that should be preserved
        in the result.
    :return: a pair ``(ordinal, tensor)``
    :rtype: tuple of frozenset and torch.Tensor
    """
    # Group sum dims by ordinal.
    dim_to_ordinal = {}
    for t, terms in tensor_tree.items():
        for term in terms:
            for dim in sum_dims.intersection(term._pyro_dims):
                dim_to_ordinal[dim] = dim_to_ordinal.get(dim, t) & t
    dims_tree = defaultdict(set)
    for dim, t in dim_to_ordinal.items():
        dims_tree[t].add(dim)

    # Recursively combine terms in different plate contexts.
    local_terms = []
    local_dims = target_dims.copy()
    local_ordinal = frozenset()
    min_ordinal = frozenset.intersection(*tensor_tree)
    while any(dims_tree.values()):
        # Arbitrarily deterministically choose a leaf.
        leaf = max(tensor_tree, key=len)
        leaf_terms = tensor_tree.pop(leaf)
        leaf_dims = dims_tree.pop(leaf, set())

        # Split terms at the current ordinal into connected components.
        for terms, dims in _partition_terms(ring, leaf_terms, leaf_dims):

            # Eliminate sum dims via a sumproduct contraction.
            term = ring.sumproduct(terms, dims - local_dims)

            # Eliminate extra plate dims via product contractions.
            if leaf == min_ordinal:
                parent = leaf
            else:
                pending_dims = sum_dims.intersection(term._pyro_dims)
                parent = frozenset.union(*(t for t, d in dims_tree.items() if d & pending_dims))
                _check_tree_structure(parent, leaf)
                contract_frames = leaf - parent
                contract_dims = dims & local_dims
                if contract_dims:
                    term, local_term = ring.global_local(term, contract_dims, contract_frames)
                    local_terms.append(local_term)
                    local_dims |= sum_dims.intersection(local_term._pyro_dims)
                    local_ordinal |= leaf
                else:
                    term = ring.product(term, contract_frames)
            tensor_tree.setdefault(parent, []).append(term)

    # Extract single tensor at root ordinal.
    assert len(tensor_tree) == 1
    ordinal, (term,) = tensor_tree.popitem()
    assert ordinal == min_ordinal

    # Perform optional localizing pass.
    if local_terms:
        assert target_dims
        local_terms.append(term)
        term = ring.sumproduct(local_terms, local_dims - target_dims)
        ordinal |= local_ordinal

    return ordinal, term


def contract_tensor_tree(tensor_tree, sum_dims, cache=None, ring=None):
    """
    Contract out ``sum_dims`` in a tree of tensors via message passing.
    This partially contracts out plate dimensions.

    This function should be deterministic and free of side effects.

    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param set sum_dims: the complete set of sum-contractions dimensions
        (indexed from the right). This is needed to distinguish sum-contraction
        dimensions from product-contraction dimensions.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :param pyro.ops.rings.Ring ring: an optional algebraic ring defining tensor
        operations.
    :returns: A contracted version of ``tensor_tree``
    :rtype: OrderedDict
    """
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, set)

    if ring is None:
        ring = LogRing(cache)

    ordinals = {term: t for t, terms in tensor_tree.items() for term in terms}
    all_terms = [term for terms in tensor_tree.values() for term in terms]
    contracted_tree = OrderedDict()

    # Split this tensor tree into connected components.
    for terms, dims in _partition_terms(ring, all_terms, sum_dims):
        component = OrderedDict()
        for term in terms:
            component.setdefault(ordinals[term], []).append(term)

        # Contract this connected component down to a single tensor.
        ordinal, term = _contract_component(ring, component, dims, set())
        contracted_tree.setdefault(ordinal, []).append(term)

    return contracted_tree


def contract_to_tensor(tensor_tree, sum_dims, target_ordinal=None, target_dims=None,
                       cache=None, ring=None):
    """
    Contract out ``sum_dims`` in a tree of tensors, via message
    passing. This reduces all terms down to a single tensor in the plate
    context specified by ``target_ordinal``, optionally preserving sum
    dimensions ``target_dims``.

    This function should be deterministic and free of side effects.

    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param set sum_dims: the complete set of sum-contractions dimensions
        (indexed from the right). This is needed to distinguish sum-contraction
        dimensions from product-contraction dimensions.
    :param frozenset target_ordinal: An optional ordinal to which the result
        will be contracted or broadcasted.
    :param set target_dims: An optional subset of ``sum_dims`` that should be
        preserved in the result.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :param pyro.ops.rings.Ring ring: an optional algebraic ring defining tensor
        operations.
    :returns: a single tensor
    :rtype: torch.Tensor
    """
    if target_ordinal is None:
        target_ordinal = frozenset()
    if target_dims is None:
        target_dims = set()
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, set)
    assert isinstance(target_ordinal, frozenset)
    assert isinstance(target_dims, set) and target_dims <= sum_dims
    if ring is None:
        ring = LogRing(cache)

    ordinals = {term: t for t, terms in tensor_tree.items() for term in terms}
    all_terms = [term for terms in tensor_tree.values() for term in terms]
    contracted_terms = []

    # Split this tensor tree into connected components.
    modulo_total = bool(target_dims)
    for terms, dims in _partition_terms(ring, all_terms, sum_dims):
        if modulo_total and dims.isdisjoint(target_dims):
            continue
        component = OrderedDict()
        for term in terms:
            component.setdefault(ordinals[term], []).append(term)

        # Contract this connected component down to a single tensor.
        ordinal, term = _contract_component(ring, component, dims, target_dims & dims)
        _check_plates_are_sensible(target_dims.intersection(term._pyro_dims),
                                   ordinal - target_ordinal)

        # Eliminate extra plate dims via product contractions.
        contract_frames = ordinal - target_ordinal
        if contract_frames:
            assert not sum_dims.intersection(term._pyro_dims)
            term = ring.product(term, contract_frames)

        contracted_terms.append(term)

    # Combine contracted tensors via product, then broadcast.
    term = ring.sumproduct(contracted_terms, set())
    assert sum_dims.intersection(term._pyro_dims) <= target_dims
    return ring.broadcast(term, target_ordinal)


def einsum(equation, *operands, **kwargs):
    """
    Generalized plated sum-product algorithm via tensor variable elimination.

    This generalizes :func:`~pyro.ops.einsum.contract` in two ways:

    1.  Multiple outputs are allowed, and intermediate results can be shared.
    2.  Inputs and outputs can be plated along symbols given in ``plates``;
        reductions along ``plates`` are product reductions.

    The best way to understand this function is to try the examples below,
    which show how :func:`einsum` calls can be implemented as multiple calls
    to :func:`~pyro.ops.einsum.contract` (which is generally more expensive).

    To illustrate multiple outputs, note that the following are equivalent::

        z1, z2, z3 = einsum('ab,bc->a,b,c', x, y)  # multiple outputs

        z1 = contract('ab,bc->a', x, y)
        z2 = contract('ab,bc->b', x, y)
        z3 = contract('ab,bc->c', x, y)

    To illustrate plated inputs, note that the following are equivalent::

        assert len(x) == 3 and len(y) == 3
        z = einsum('ab,ai,bi->b', w, x, y, plates='i')

        z = contract('ab,a,a,a,b,b,b->b', w, *x, *y)

    When a sum dimension `a` always appears with a plate dimension `i`,
    then `a` corresponds to a distinct symbol for each slice of `a`. Thus
    the following are equivalent::

        assert len(x) == 3 and len(y) == 3
        z = einsum('ai,ai->', x, y, plates='i')

        z = contract('a,b,c,a,b,c->', *x, *y)

    When such a sum dimension appears in the output, it must be
    accompanied by all of its plate dimensions, e.g. the following are
    equivalent::

        assert len(x) == 3 and len(y) == 3
        z = einsum('abi,abi->bi', x, y, plates='i')

        z0 = contract('ab,ac,ad,ab,ac,ad->b', *x, *y)
        z1 = contract('ab,ac,ad,ab,ac,ad->c', *x, *y)
        z2 = contract('ab,ac,ad,ab,ac,ad->d', *x, *y)
        z = torch.stack([z0, z1, z2])

    Note that each plate slice through the output is multilinear in all plate
    slices through all inptus, thus e.g. batch matrix multiply would be
    implemented *without* ``plates``, so the following are all equivalent::

        xy = einsum('abc,acd->abd', x, y, plates='')
        xy = torch.stack([xa.mm(ya) for xa, ya in zip(x, y)])
        xy = torch.bmm(x, y)

    Among all valid equations, some computations are polynomial in the sizes of
    the input tensors and other computations are exponential in the sizes of
    the input tensors. This function raises :py:class:`NotImplementedError`
    whenever the computation is exponential.

    :param str equation: An einsum equation, optionally with multiple outputs.
    :param torch.Tensor operands: A collection of tensors.
    :param str plates: An optional string of plate symbols.
    :param str backend: An optional einsum backend, defaults to 'torch'.
    :param dict cache: An optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :param bool modulo_total: Optionally allow einsum to arbitrarily scale
        each result plate, which can significantly reduce computation. This is
        safe to set whenever each result plate denotes a nonnormalized
        probability distribution whose total is not of interest.
    :return: a tuple of tensors of requested shape, one entry per output.
    :rtype: tuple
    :raises ValueError: if tensor sizes mismatch or an output requests a
        plated dim without that dim's plates.
    :raises NotImplementedError: if contraction would have cost exponential in
        the size of any input tensor.
    """
    # Extract kwargs.
    cache = kwargs.pop('cache', None)
    plates = kwargs.pop('plates', '')
    backend = kwargs.pop('backend', 'torch')
    modulo_total = kwargs.pop('modulo_total', False)
    try:
        Ring = BACKEND_TO_RING[backend]
    except KeyError as e:
        raise NotImplementedError('\n'.join(
            ['Only the following pyro backends are currently implemented:'] +
            list(BACKEND_TO_RING))) from e

    # Parse generalized einsum equation.
    if '.' in equation:
        raise NotImplementedError('ubsersum does not yet support ellipsis notation')
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    outputs = outputs.split(',')
    assert len(inputs) == len(operands)
    assert all(isinstance(x, torch.Tensor) for x in operands)
    if not modulo_total and any(outputs):
        raise NotImplementedError('Try setting modulo_total=True and ensuring that your use case '
                                  'allows an arbitrary scale factor on each result plate.')
    if len(operands) != len(set(operands)):
        operands = [x[...] for x in operands]  # ensure tensors are unique

    # Check sizes.
    with ignore_jit_warnings():
        dim_to_size = {}
        for dims, term in zip(inputs, operands):
            for dim, size in zip(dims, map(int, term.shape)):
                old = dim_to_size.setdefault(dim, size)
                if old != size:
                    raise ValueError(u"Dimension size mismatch at dim '{}': {} vs {}"
                                     .format(dim, size, old))

    # Construct a tensor tree shared by all outputs.
    tensor_tree = OrderedDict()
    plates = frozenset(plates)
    for dims, term in zip(inputs, operands):
        assert len(dims) == term.dim()
        term._pyro_dims = dims
        ordinal = plates.intersection(dims)
        tensor_tree.setdefault(ordinal, []).append(term)

    # Compute outputs, sharing intermediate computations.
    results = []
    with shared_intermediates(cache) as cache:
        ring = Ring(cache, dim_to_size=dim_to_size)
        for output in outputs:
            sum_dims = set(output).union(*inputs) - set(plates)
            term = contract_to_tensor(tensor_tree, sum_dims,
                                      target_ordinal=plates.intersection(output),
                                      target_dims=sum_dims.intersection(output),
                                      ring=ring)
            if term._pyro_dims != output:
                term = term.permute(*map(term._pyro_dims.index, output))
                term._pyro_dims = output
            results.append(term)
    return tuple(results)


def ubersum(equation, *operands, **kwargs):
    """
    Deprecated, use :func:`einsum` instead.
    """
    warnings.warn("'ubersum' is deprecated, use 'pyro.ops.contract.einsum' instead",
                  DeprecationWarning)
    if 'batch_dims' in kwargs:
        warnings.warn("'batch_dims' is deprecated, use 'plates' instead",
                      DeprecationWarning)
        kwargs['plates'] = kwargs.pop('batch_dims')
    kwargs.setdefault('backend', 'pyro.ops.einsum.torch_log')
    return einsum(equation, *operands, **kwargs)


def _select(tensor, dims, indices):
    for dim, index in zip(dims, indices):
        tensor = tensor.select(dim, index)
    return tensor


class _DimUnroller:
    """
    Object to map plated dims to collections of unrolled dims.

    :param dict dim_to_ordinal: a mapping from contraction dim to the set of
        plates over which the contraction dim is plated.
    """
    def __init__(self, dim_to_ordinal):
        self._plates = {d: tuple(sorted(ordinal)) for d, ordinal in dim_to_ordinal.items()}
        self._symbols = map(opt_einsum.get_symbol, itertools.count())
        self._map = {}

    def __call__(self, dim, indices):
        """
        Converts a plate dim + plate indices to a unrolled dim.

        :param str dim: a plate dimension to unroll
        :param dict indices: a mapping from plate dimension to int
        :return: a unrolled dim
        :rtype: str
        """
        plate = self._plates.get(dim, ())
        index = tuple(indices[d] for d in plate)
        key = dim, index
        if key in self._map:
            return self._map[key]
        normal_dim = next(self._symbols)
        self._map[key] = normal_dim
        return normal_dim


def naive_ubersum(equation, *operands, **kwargs):
    """
    Naive reference implementation of :func:`ubersum` via unrolling.

    This implementation should never raise ``NotImplementedError``.
    This implementation should agree with :func:`ubersum` whenver
    :func:`ubersum` does not raise ``NotImplementedError``.
    """
    # Parse equation, without loss of generality assuming a single output.
    inputs, outputs = equation.split('->')
    outputs = outputs.split(',')
    if len(outputs) > 1:
        return tuple(naive_ubersum(inputs + '->' + output, *operands, **kwargs)[0]
                     for output in outputs)
    output, = outputs
    inputs = inputs.split(',')
    backend = kwargs.pop('backend', 'pyro.ops.einsum.torch_log')

    # Split dims into plate dims, contraction dims, and dims to keep.
    plates = set(kwargs.pop('plates', ''))
    if not plates:
        result = opt_einsum.contract(equation, *operands, backend=backend)
        return (result,)
    output_dims = set(output)

    # Collect sizes of all dimensions.
    sizes = {}
    for input_, operand in zip(inputs, operands):
        for dim, size in zip(input_, operand.shape):
            old = sizes.setdefault(dim, size)
            if old != size:
                raise ValueError(u"Dimension size mismatch at dim '{}': {} vs {}"
                                 .format(dim, size, old))

    # Compute plate context for each non-plate dim, by convention the
    # intersection over all plate contexts of tensors in which the dim appears.
    dim_to_ordinal = {}
    for dims in map(set, inputs):
        ordinal = dims & plates
        for dim in dims - plates:
            dim_to_ordinal[dim] = dim_to_ordinal.get(dim, ordinal) & ordinal
    for dim in output_dims - plates:
        _check_plates_are_sensible({dim}, dim_to_ordinal[dim] - output_dims)

    # Unroll by replicating along plate dimensions.
    unroll_dim = _DimUnroller(dim_to_ordinal)
    flat_inputs = []
    flat_operands = []
    for input_, operand in zip(inputs, operands):
        local_dims = [d for d in input_ if d in plates]
        offsets = [input_.index(d) - len(input_) for d in local_dims]
        for index in itertools.product(*(range(sizes[d]) for d in local_dims)):
            flat_inputs.append(''.join(unroll_dim(d, dict(zip(local_dims, index)))
                                       for d in input_ if d not in plates))
            flat_operands.append(_select(operand, offsets, index))

    # Defer to unplated einsum.
    result = torch.empty(torch.Size(sizes[d] for d in output),
                         dtype=operands[0].dtype, device=operands[0].device)
    local_dims = [d for d in output if d in plates]
    offsets = [output.index(d) - len(output) for d in local_dims]
    for index in itertools.product(*(range(sizes[d]) for d in local_dims)):
        flat_output = ''.join(unroll_dim(d, dict(zip(local_dims, index)))
                              for d in output if d not in plates)
        flat_equation = ','.join(flat_inputs) + '->' + flat_output
        flat_result = opt_einsum.contract(flat_equation, *flat_operands, backend=backend)
        if not local_dims:
            result = flat_result
            break
        _select(result, offsets, index).copy_(flat_result)
    return (result,)
