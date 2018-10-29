from __future__ import absolute_import, division, print_function

import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import numpy as np
import opt_einsum
import torch
from opt_einsum import shared_intermediates
from six import add_metaclass
from six.moves import map

from pyro.distributions.util import broadcast_shape
from pyro.ops.einsum import contract
from pyro.ops.sumproduct import logsumproductexp


def _finfo(tensor):
    # This can be replaced with torch.finfo once it is available
    # https://github.com/pytorch/pytorch/issues/10742
    return np.finfo(torch.empty(torch.Size(), dtype=tensor.dtype).numpy().dtype)


@add_metaclass(ABCMeta)
class TensorRing(object):
    """
    Abstract tensor ring class.

    Each tensor ring class has a notion of ``dims`` that can be sum-contracted
    out, and a notion of ``ordinal`` that represents a set of batch dimensions
    that can be broadcasted-up or product-contracted out.
    Implementations should cache intermediate results to be compatible with
    :func:`~opt_einsum.shared_intermediates`.

    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    """
    def __init__(self, cache=None):
        self._cache = {} if cache is None else cache

    def _save_tensor(self, tensor):
        """
        Saves a tensor in the cache so that ``id(tensor)`` can be used as a
        key in the cache without risk if the id being recycled.
        """
        self._cache['tensor', id(tensor)] = tensor

    @abstractmethod
    def dims(self, term):
        """
        Returns an iterable of nontrivial dims associted with this term.
        Derived classes may use any hashable type for dims.
        """
        raise NotImplementedError

    @abstractmethod
    def sumproduct(self, terms, dims):
        """
        Multiply all ``terms`` together, then sum-contract out all ``dims``
        from the result.

        :param list terms: a list of tensors
        :param dims: an iterable of sum dims to contract
        """
        raise NotImplementedError

    @abstractmethod
    def product(self, term, ordinal):
        """
        Product-contract the given ``term`` along any batch dimensions
        present in given ``ordinal``.

        :param torch.Tensor term: the term to contract
        :param frozenset ordinal: an ordinal specifying batch dims to contract
        """
        raise NotImplementedError

    @abstractmethod
    def broadcast(self, tensor, ordinal):
        """
        Broadcast the given ``term`` by expanding along any batch dimensions
        present in ``ordinal`` but not ``term``.

        :param torch.Tensor term: the term to expand
        :param frozenset ordinal: an ordinal specifying batch context
        """
        raise NotImplementedError

    def inv(self, term):
        """
        Computes the reciprocal of a term, for use in inclusion-exclusion.

        The default implementation assumes log-space representation.

        :param torch.Tensor term: the term to invert
        """
        key = 'inv', id(term)
        if key not in self._cache:
            value = -term
            value.clamp_(max=_finfo(value).max)  # avoid nan due to inf - inf
            self._cache['dims', id(value)] = self.dims(term)
            self._cache[key] = value
        return self._cache[key]

    def inclusion_exclusion(self, term, dims, ordinal):
        """
        Computes the inclusion-exclusion message for tensor message passing::

            term / sum(term, dims) * product(sum(term, dims), ordinal)

        :param torch.Tensor term: the term to contract
        :param dims: an iterable of sum dims to contract
        :param frozenset ordinal: an ordinal specifying batch dims to contract
        """
        if not dims:
            return self.product(term, ordinal)
        key = 'inclusion_exclusion', id(term), frozenset(dims), ordinal
        if key not in self._cache:
            term_sum = self.sumproduct([term], dims)
            term_sum_prod = self.product(term_sum, ordinal)
            value = self.sumproduct([term, self.inv(term_sum), term_sum_prod], set())
            assert self.dims(value) == self.dims(term)
            self._cache[key] = value
        return self._cache[key]


class UnpackedLogRing(TensorRing):
    """
    Tensor Ring defined by high-dimensional unpacked tensors in log space.

    Tensor values are in log units, so ``sum`` is implemented as ``logsumexp``,
    and ``product`` is implemented as ``sum``.
    Tensor shapes are typically wide with only a few nontrivial dimensions::

        torch.Size((7, 1, 1, 1, 1, 1, 3, 1, 1, 2))

    Dims are negative integers indexing into tensors shapes from the right.
    Ordinals are frozensets of ``CondIndepStackFrame``s.
    """
    def dims(self, term):
        key = 'dims', id(term)
        if key not in self._cache:
            self._cache[key] = tuple(d for d in range(-term.dim(), 0) if term.size(d) > 1)
        return self._cache[key]

    def sumproduct(self, terms, dims):
        key = 'sumproduct', frozenset(id(x) for x in terms), frozenset(dims)
        if key in self._cache:
            return self._cache[key]

        if dims:
            assert all(dim < 0 for dim in dims)
            shape = list(broadcast_shape(*set(x.shape for x in terms)))
            for dim in dims:
                shape[dim] = 1
            term = logsumproductexp(terms, tuple(shape))
        else:
            term = sum(terms)

        # Aggressively squeeze to improve sharing.
        while term.dim() and term.size(0) == 1:
            term = term.squeeze(0)
        self._save_tensor(term)
        self._cache[key] = term
        return term

    def product(self, term, ordinal):
        for frame in sorted(ordinal, key=lambda f: -f.dim):
            if -frame.dim <= term.dim() and term.size(frame.dim) != 1:
                key = 'product', id(term), frame.dim
                if key in self._cache:
                    term = self._cache[key]
                else:
                    self._save_tensor(term)
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
        key = 'broadcast', id(term), shape
        if key in self._cache:
            return self._cache[key]
        self._save_tensor(term)
        term = term.expand(shape)
        self._cache[key] = term
        return term


class PackedLogRing(TensorRing):
    """
    Tensor Ring of packed tensors with named dimensions in log space.

    Tensor values are in log units, so ``sum`` is implemented as ``logsumexp``,
    and ``product`` is implemented as ``sum``.
    Tensor dimensions are packed; to read the name of a tensor, call
    :meth:`dims`, which returns a string of dimension names aligned with the
    tensor's shape.

    Dims are characters (string or unicode).
    Ordinals are frozensets of characters.
    """
    def __init__(self, inputs, operands, cache=None):
        super(PackedLogRing, self).__init__(cache=cache)
        self._batch_size = {}
        for dims, term in zip(inputs, operands):
            self._save_tensor(term)
            self._cache['dims', id(term)] = dims
            for dim, size in zip(dims, term.shape):
                old = self._batch_size.setdefault(dim, size)
                if old != size:
                    raise ValueError(u"Dimension size mismatch at dim '{}': {} vs {}"
                                     .format(dim, size, old))

    def dims(self, term):
        return self._cache['dims', id(term)]

    def sumproduct(self, terms, dims):
        inputs = [self.dims(term) for term in terms]
        output = ''.join(sorted(set(''.join(inputs)) - set(dims)))
        equation = ','.join(inputs) + '->' + output
        term = contract(equation, *terms, backend='pyro.ops.einsum.torch_log')
        self._save_tensor(term)
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
                    self._save_tensor(term)
                    term = term.sum(pos)
                    dims = dims.replace(dim, '')
                    self._cache[key] = term
                    self._cache['dims', id(term)] = dims
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


def _partition_terms(ring, terms, dims):
    """
    Given a list of terms and a set of contraction dims, partitions the terms
    up into sets that must be contracted together. By separating these
    components we avoid broadcasting. This function should be deterministic.

    This function should be deterministic and free of side effects.
    """
    # Construct a bipartite graph between terms and the dims in which they
    # are enumerated. This conflates terms and dims (tensors and ints).
    neighbors = OrderedDict([(t, []) for t in terms] + [(d, []) for d in sorted(dims)])
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
        if component_terms:
            component_dims = set(v for v in component if not isinstance(v, torch.Tensor))
            yield component_terms, component_dims


def _contract_component(ring, tensor_tree, sum_dims, target_dims):
    """
    Contract out ``sum_dims`` in a tree of tensors in-place, via message
    passing. This reduces all tensors down to a single tensor in the greatest
    lower bound plate context.

    This function should be deterministic.
    This function has side-effects: it modifies ``tensor_tree``, ``sum_dims``,
    and ``target_dims`` in-place.

    :param TensorRing ring: an algebraic ring defining tensor operations.
    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param set sum_dims: the complete set of sum-contractions dimensions
        (indexed from the right). This is needed to distinguish sum-contraction
        dimensions from product-contraction dimensions.
    :param set target_dims: A subset of ``sum_dims`` that should be preserved
        in the result.
    """
    assert target_dims <= sum_dims

    # First close the set of ordinals under intersection (greatest lower bound),
    # ensuring that the ordinals are arranged in a tree structure.
    target_ordinal = frozenset.intersection(*tensor_tree)
    tensor_tree.setdefault(target_ordinal, [])
    pending = list(tensor_tree)
    while pending:
        t = pending.pop()
        for u in list(tensor_tree):
            tu = t & u
            if tu not in tensor_tree:
                tensor_tree[tu] = []
                pending.append(tu)

    # Collect contraction dimensions by ordinal.
    dim_to_ordinal = {}
    for t, terms in tensor_tree.items():
        for term in terms:
            for dim in ring.dims(term):
                if dim in sum_dims:
                    dim_to_ordinal[dim] = dim_to_ordinal.get(dim, t) & t
    dims_tree = defaultdict(set)
    for dim, t in dim_to_ordinal.items():
        dims_tree[t].add(dim)

    # Recursively combine terms in different plate contexts.
    while any(dims_tree.values()):
        leaf = max(tensor_tree, key=len)
        leaf_terms = tensor_tree.pop(leaf)
        leaf_dims = dims_tree.pop(leaf, set())

        # Split terms at the current ordinal into connected components.
        for terms, dims in _partition_terms(ring, leaf_terms, leaf_dims):

            # Eliminate sum dims via a sumproduct contraction.
            term = ring.sumproduct(terms, dims - target_dims)
            pending_dims = sum_dims.intersection(ring.dims(term))

            # Eliminate extra plate dims via product contractions.
            if leaf == target_ordinal:
                parent = leaf
            else:
                parent = frozenset.union(*(t for t, d in dims_tree.items() if d & pending_dims))
                if parent == leaf:
                    raise NotImplementedError(
                        "Expected tree-structured plate nesting, but found "
                        "dependencies on independent plates [{}]. "
                        "Try converting one of the plates to an irange (but beware "
                        "exponential cost in the size of that irange)"
                        .format(', '.join(getattr(f, 'name', str(f)) for f in leaf)))
                contract_frames = leaf - parent
                term = ring.inclusion_exclusion(term, dims & target_dims, contract_frames)

            tensor_tree.setdefault(parent, []).append(term)


def contract_tensor_tree(tensor_tree, sum_dims, target_dims=None, ring=None, cache=None):
    """
    Contract out ``sum_dims`` in a tree of tensors via message passing.

    This function should be deterministic and free of side effects.

    :param OrderedDict tensor_tree: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param set sum_dims: the complete set of sum-contractions dimensions
        (indexed from the right). This is needed to distinguish sum-contraction
        dimensions from product-contraction dimensions.
    :param set target_dims: An optional subset of ``sum_dims`` that should be
        preserved in the result.
    :param TensorRing ring: an algebraic ring defining tensor operations.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :returns: A contracted version of ``tensor_tree``
    :rtype: OrderedDict
    """
    if target_dims is None:
        target_dims = set()
    if ring is None:
        ring = UnpackedLogRing(cache=cache)
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, set)
    assert isinstance(target_dims, set) and target_dims <= sum_dims
    assert isinstance(ring, TensorRing)

    ordinals = {term: t for t, terms in tensor_tree.items() for term in terms}
    all_terms = [term for terms in tensor_tree.values() for term in terms]
    contracted_tree = OrderedDict()

    # Split this tensor tree into connected components.
    for terms, dims in _partition_terms(ring, all_terms, sum_dims):
        component = OrderedDict()
        for term in terms:
            component.setdefault(ordinals[term], []).append(term)

        # Contract this connected component down to a single tensor.
        _contract_component(ring, component, dims, target_dims & dims)
        assert len(component) == 1
        t, terms = component.popitem()
        assert len(terms) == 1
        contracted_tree.setdefault(t, []).extend(terms)

    return contracted_tree


def contract_to_tensor(tensor_tree, sum_dims, target_ordinal=None, target_dims=None, ring=None, cache=None):
    """
    Contract out ``sum_dims`` in a tree of tensors, via message
    passing. This reduces all terms down to a single tensor in the plate
    context specified by ``target_ordinal``, preserving dimensions
    ``target_dims``.

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
    :param TensorRing ring: an algebraic ring defining tensor operations.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :returns: a single tensor
    :rtype: torch.Tensor
    """
    if target_ordinal is None:
        target_ordinal = frozenset()
    if target_dims is None:
        target_dims = set()
    if ring is None:
        ring = UnpackedLogRing(cache=cache)
    assert isinstance(tensor_tree, OrderedDict)
    assert isinstance(sum_dims, set)
    assert isinstance(target_ordinal, frozenset)
    assert isinstance(target_dims, set) and target_dims <= sum_dims
    assert isinstance(ring, TensorRing)

    # Contract out all sum dims via sumproduct contractions.
    tensor_tree = contract_tensor_tree(tensor_tree, sum_dims, target_dims, ring=ring)

    # Eliminate extra plate dims via product contractions.
    lower_terms = []
    lower_ordinal = frozenset()
    for ordinal, terms in tensor_tree.items():
        contract_frames = ordinal - target_ordinal
        if contract_frames:
            for term in terms:
                bad_dims = target_dims.intersection(ring.dims(term))
                if bad_dims:
                    raise ValueError(u"It is nonsensical to preserve a batched dim without preserving "
                                     u"all of that dim's batch dims, but found '{}' without '{}'"
                                     .format(bad_dims, ','.join(contract_frames)))
            ordinal = ordinal & target_ordinal
            terms = [ring.product(term, contract_frames) for term in terms]
        lower_terms.extend(terms)
        lower_ordinal = lower_ordinal | ordinal
    assert lower_ordinal <= target_ordinal

    # Combine and broadcast terms.
    lower_term = ring.sumproduct(lower_terms, set())
    return ring.broadcast(lower_term, target_ordinal)


def ubersum(equation, *operands, **kwargs):
    """
    Generalized batched sum-product algorithm via tensor message passing.

    This generalizes :func:`~pyro.ops.einsum.contract` in two ways:

    1.  Multiple outputs are allowed, and intermediate results can be shared.
    2.  Inputs and outputs can be batched along symbols given in ``batch_dims``;
        reductions along ``batch_dims`` are product reductions.

    The best way to understand this function is to try the examples below,
    which show how :func:`ubersum` calls can be implemented as multiple calls
    to :func:`~pyro.ops.einsum.contract` (which is generally more expensive).

    To illustrate multiple outputs, note that the following are equivalent::

        z1, z2, z3 = ubersum('ab,bc->a,b,c', x, y)  # multiple outputs

        backend = 'pyro.ops.einsum.torch_log'
        z1 = contract('ab,bc->a', x, y, backend=backend)
        z2 = contract('ab,bc->b', x, y, backend=backend)
        z3 = contract('ab,bc->c', x, y, backend=backend)

    To illustrate batched inputs, note that the following are equivalent::

        assert len(x) == 3 and len(y) == 3
        z = ubersum('ab,ai,bi->b', w, x, y, batch_dims='i')

        z = contract('ab,a,a,a,b,b,b->b', w, *x, *y, backend=backend)

    When a sum dimension `a` always appears with a batch dimension `i`,
    then `a` corresponds to a distinct symbol for each slice of `a`. Thus
    the following are equivalent::

        assert len(x) == 3 and len(y) == 3
        z = ubersum('ai,ai->', x, y, batch_dims='i')

        z = contract('a,b,c,a,b,c->', *x, *y, backend=backend)

    When such a sum dimension appears in the output, it must be
    accompanied by all of its batch dimensions, e.g. the following are
    equivalent::

        assert len(x) == 3 and len(y) == 3
        z = ubersum('abi,abi->bi', x, y, batch_dims='i')

        z0 = contract('ab,ac,ad,ab,ac,ad->b', *x, *y, backend=backend)
        z1 = contract('ab,ac,ad,ab,ac,ad->c', *x, *y, backend=backend)
        z2 = contract('ab,ac,ad,ab,ac,ad->d', *x, *y, backend=backend)
        z = torch.stack([z0, z1, z2])

    Among all valid equations, some computations are polynomial in the sizes of
    the input tensors and other computations are exponential in the sizes of
    the input tensors. This function raises :py:class:`NotImplementedError`
    whenever the computation is exponential.

    :param str equation: an einsum equation, optionally with multiple outputs.
    :param torch.Tensor operands: a collection of tensors
    :param str batch_dims: an optional string of batch dims.
    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    :return: a tuple of tensors of requested shape, one entry per output.
    :rtype: tuple
    :raises ValueError: if tensor sizes mismatch or an output requests a
        batched dim without that dim's batch dims.
    :raises NotImplementedError: if contraction would have cost exponential in
        the size of any input tensor.
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
    assert len(inputs) == len(operands)
    assert all(isinstance(x, torch.Tensor) for x in operands)
    if len(operands) != len(set(operands)):
        operands = [x[...] for x in operands]  # ensure tensors are unique

    # Construct a tensor tree shared by all outputs.
    tensor_tree = OrderedDict()
    batch_dims = frozenset(batch_dims)
    for dims, term in zip(inputs, operands):
        assert len(dims) == term.dim()
        ordinal = batch_dims.intersection(dims)
        tensor_tree.setdefault(ordinal, []).append(term)

    # Compute outputs, sharing intermediate computations.
    results = []
    with shared_intermediates(cache) as cache:
        ring = PackedLogRing(inputs, operands, cache=cache)
        for output in outputs:
            sum_dims = set(output).union(*inputs) - set(batch_dims)
            term = contract_to_tensor(tensor_tree, sum_dims,
                                      target_ordinal=batch_dims.intersection(output),
                                      target_dims=sum_dims.intersection(output),
                                      ring=ring)
            dims = ring.dims(term)
            if dims != output:
                term = term.permute(*map(dims.index, output))
            results.append(term)
    return tuple(results)


def _select(tensor, dims, indices):
    for dim, index in zip(dims, indices):
        tensor = tensor.select(dim, index)
    return tensor


class _DimFlattener(object):
    """
    Object to map batched dims to batches of flat dims.

    :param dict dim_to_ordinal: a mapping from contraction dim to the set of
        batch dims over which the contraction dim is batched.
    """
    def __init__(self, dim_to_ordinal):
        self._plates = {d: tuple(sorted(ordinal)) for d, ordinal in dim_to_ordinal.items()}
        self._symbols = map(opt_einsum.get_symbol, itertools.count())
        self._map = {}

    def __call__(self, dim, indices):
        """
        Converts a batched dim + batch indices to a flattened dim.

        :param str dim: a batched dimension to flatten
        :param dict indices: a mapping from batch dimension to int
        :return: a flattened dim
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
    Naive reference implementation of :func:`ubersum`.

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

    # Split dims into batch dims, contraction dims, and dims to keep.
    batch_dims = set(kwargs.pop('batch_dims', ''))
    if not batch_dims:
        result = opt_einsum.contract(equation, *operands, backend='pyro.ops.einsum.torch_log')
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

    # Compute batch context for each non-batch dim, by convention the
    # intersection over all batch contexts of tensors in which the dim appears.
    dim_to_ordinal = {}
    for dims in map(set, inputs):
        ordinal = dims & batch_dims
        for dim in dims - batch_dims:
            dim_to_ordinal[dim] = dim_to_ordinal.get(dim, ordinal) & ordinal
    for dim in output_dims - batch_dims:
        missing_dims = dim_to_ordinal[dim] - output_dims
        if missing_dims:
            raise ValueError(u"It is nonsensical to preserve a batched dim without preserving "
                             u"all of that dim's batch dims, but found '{}' without '{}' in '{}'"
                             .format(dim, ','.join(missing_dims), equation))

    # Flatten by replicating along batch dimensions.
    flatten_dim = _DimFlattener(dim_to_ordinal)
    flat_inputs = []
    flat_operands = []
    for input_, operand in zip(inputs, operands):
        local_dims = [d for d in input_ if d in batch_dims]
        offsets = [input_.index(d) - len(input_) for d in local_dims]
        for index in itertools.product(*(range(sizes[d]) for d in local_dims)):
            flat_inputs.append(''.join(flatten_dim(d, dict(zip(local_dims, index)))
                                       for d in input_ if d not in batch_dims))
            flat_operands.append(_select(operand, offsets, index))

    # Defer to unbatched einsum.
    result = operands[0].new_empty(torch.Size(sizes[d] for d in output))
    local_dims = [d for d in output if d in batch_dims]
    offsets = [output.index(d) - len(output) for d in local_dims]
    for index in itertools.product(*(range(sizes[d]) for d in local_dims)):
        flat_output = ''.join(flatten_dim(d, dict(zip(local_dims, index)))
                              for d in output if d not in batch_dims)
        flat_equation = ','.join(flat_inputs) + '->' + flat_output
        flat_result = opt_einsum.contract(flat_equation, *flat_operands,
                                          backend='pyro.ops.einsum.torch_log')
        _select(result, offsets, index).copy_(flat_result)
    return (result,)
