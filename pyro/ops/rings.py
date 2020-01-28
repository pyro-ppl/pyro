# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import weakref
from abc import ABCMeta, abstractmethod

import torch

from pyro.ops.einsum import contract
from pyro.ops.einsum.adjoint import SAMPLE_SYMBOL, Backward
from pyro.util import ignore_jit_warnings


class Ring(object, metaclass=ABCMeta):
    """
    Abstract tensor ring class.

    Each tensor ring class has a notion of ``dims`` that can be sum-contracted
    out, and a notion of ``ordinal`` that represents a set of plate dimensions
    that can be broadcasted-up or product-contracted out.
    Implementations should cache intermediate results to be compatible with
    :func:`~opt_einsum.shared_intermediates`.

    Dims are characters (string or unicode).
    Ordinals are frozensets of characters.

    :param dict cache: an optional :func:`~opt_einsum.shared_intermediates`
        cache.
    """
    def __init__(self, cache=None):
        self._cache = {} if cache is None else cache

    def _hash_by_id(self, tensor):
        """
        Returns the id of a tensor and saves the tensor so that this id can be
        used as a key in the cache without risk of the id being recycled.
        """
        result = id(tensor)
        assert self._cache.setdefault(('tensor', result), tensor) is tensor
        return result

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
        Product-contract the given ``term`` along any plate dimensions
        present in given ``ordinal``.

        :param torch.Tensor term: the term to contract
        :param frozenset ordinal: an ordinal specifying plates to contract
        """
        raise NotImplementedError

    def broadcast(self, term, ordinal):
        """
        Broadcast the given ``term`` by expanding along any plate dimensions
        present in ``ordinal`` but not ``term``.

        :param torch.Tensor term: the term to expand
        :param frozenset ordinal: an ordinal specifying plates
        """
        dims = term._pyro_dims
        missing_dims = ''.join(sorted(set(ordinal) - set(dims)))
        if missing_dims:
            key = 'broadcast', self._hash_by_id(term), missing_dims
            if key in self._cache:
                term = self._cache[key]
            else:
                missing_shape = tuple(self._dim_to_size[dim] for dim in missing_dims)
                term = term.expand(missing_shape + term.shape)
                dims = missing_dims + dims
                self._cache[key] = term
                term._pyro_dims = dims
        return term

    @abstractmethod
    def inv(self, term):
        """
        Computes the reciprocal of a term, for use in inclusion-exclusion.

        :param torch.Tensor term: the term to invert
        """
        raise NotImplementedError

    def global_local(self, term, dims, ordinal):
        r"""
        Computes global and local terms for tensor message passing
        using inclusion-exclusion::

            term / sum(term, dims) * product(sum(term, dims), ordinal)
            \____________________/   \_______________________________/
                  local part                    global part

        :param torch.Tensor term: the term to contract
        :param dims: an iterable of sum dims to contract
        :param frozenset ordinal: an ordinal specifying plates to contract
        :return: a tuple ``(global_part, local_part)`` as defined above
        :rtype: tuple
        """
        assert dims, 'dims was empty, use .product() instead'
        key = 'global_local', self._hash_by_id(term), frozenset(dims), ordinal
        if key in self._cache:
            return self._cache[key]

        term_sum = self.sumproduct([term], dims)
        global_part = self.product(term_sum, ordinal)
        with ignore_jit_warnings():
            local_part = self.sumproduct([term, self.inv(term_sum)], set())
        assert sorted(local_part._pyro_dims) == sorted(term._pyro_dims)
        result = global_part, local_part
        self._cache[key] = result
        return result


class LinearRing(Ring):
    """
    Ring of sum-product operations in linear space.

    Tensor dimensions are packed; to read the name of a tensor, read the
    ``._pyro_dims`` attribute, which is a string of dimension names aligned
    with the tensor's shape.
    """
    _backend = 'torch'

    def __init__(self, cache=None, dim_to_size=None):
        super().__init__(cache=cache)
        self._dim_to_size = {} if dim_to_size is None else dim_to_size

    def sumproduct(self, terms, dims):
        inputs = [term._pyro_dims for term in terms]
        output = ''.join(sorted(set(''.join(inputs)) - set(dims)))
        equation = ','.join(inputs) + '->' + output
        term = contract(equation, *terms, backend=self._backend)
        term._pyro_dims = output
        return term

    def product(self, term, ordinal):
        dims = term._pyro_dims
        for dim in sorted(ordinal, reverse=True):
            pos = dims.find(dim)
            if pos != -1:
                key = 'product', self._hash_by_id(term), dim
                if key in self._cache:
                    term = self._cache[key]
                else:
                    term = term.prod(pos)
                    dims = dims.replace(dim, '')
                    self._cache[key] = term
                    term._pyro_dims = dims
        return term

    def inv(self, term):
        key = 'inv', self._hash_by_id(term)
        if key in self._cache:
            return self._cache[key]

        result = term.reciprocal()
        result = result.clamp(max=torch.finfo(result.dtype).max)  # avoid nan due to inf / inf
        result._pyro_dims = term._pyro_dims
        self._cache[key] = result
        return result


class LogRing(Ring):
    """
    Ring of sum-product operations in log space.

    Tensor values are in log units, so ``sum`` is implemented as ``logsumexp``,
    and ``product`` is implemented as ``sum``.
    Tensor dimensions are packed; to read the name of a tensor, read the
    ``._pyro_dims`` attribute, which is a string of dimension names aligned
    with the tensor's shape.
    """
    _backend = 'pyro.ops.einsum.torch_log'

    def __init__(self, cache=None, dim_to_size=None):
        super().__init__(cache=cache)
        self._dim_to_size = {} if dim_to_size is None else dim_to_size

    def sumproduct(self, terms, dims):
        inputs = [term._pyro_dims for term in terms]
        output = ''.join(sorted(set(''.join(inputs)) - set(dims)))
        equation = ','.join(inputs) + '->' + output
        term = contract(equation, *terms, backend=self._backend)
        term._pyro_dims = output
        return term

    def product(self, term, ordinal):
        dims = term._pyro_dims
        for dim in sorted(ordinal, reverse=True):
            pos = dims.find(dim)
            if pos != -1:
                key = 'product', self._hash_by_id(term), dim
                if key in self._cache:
                    term = self._cache[key]
                else:
                    term = term.sum(pos)
                    dims = dims.replace(dim, '')
                    self._cache[key] = term
                    term._pyro_dims = dims
        return term

    def inv(self, term):
        key = 'inv', self._hash_by_id(term)
        if key in self._cache:
            return self._cache[key]

        result = -term
        result = result.clamp(max=torch.finfo(result.dtype).max)  # avoid nan due to inf - inf
        result._pyro_dims = term._pyro_dims
        self._cache[key] = result
        return result


class _SampleProductBackward(Backward):
    """
    Backward-sample implementation of product.

    This is agnostic to sampler implementation, and hence can be used both by
    :class:`MapRing` (temperature 0 sampling) and :class:`SampleRing`
    (temperature 1 sampling).
    """
    def __init__(self, ring, term, ordinal):
        self.ring = ring
        self.term = term
        self.ordinal = ordinal

    def process(self, message):
        if message is not None:
            sample_dims = message._pyro_sample_dims
            message = self.ring.broadcast(message, self.ordinal)
            if message._pyro_dims.index(SAMPLE_SYMBOL) != 0:
                dims = SAMPLE_SYMBOL + message._pyro_dims.replace(SAMPLE_SYMBOL, '')
                message = message.permute(tuple(map(message._pyro_dims.find, dims)))
                message._pyro_dims = dims
                assert message.dim() == len(message._pyro_dims)
            message._pyro_sample_dims = sample_dims
            assert message.size(0) == len(message._pyro_sample_dims)
        yield self.term._pyro_backward, message


class MapRing(LogRing):
    """
    Ring of forward-maxsum backward-argmax operations.
    """
    _backend = 'pyro.ops.einsum.torch_map'

    def product(self, term, ordinal):
        result = super().product(term, ordinal)
        if hasattr(term, '_pyro_backward'):
            result._pyro_backward = _SampleProductBackward(self, term, ordinal)
        return result


class SampleRing(LogRing):
    """
    Ring of forward-sumproduct backward-sample operations in log space.
    """
    _backend = 'pyro.ops.einsum.torch_sample'

    def product(self, term, ordinal):
        result = super().product(term, ordinal)
        if hasattr(term, '_pyro_backward'):
            result._pyro_backward = _SampleProductBackward(self, term, ordinal)
        return result


class _MarginalProductBackward(Backward):
    """
    Backward-marginal implementation of product, using inclusion-exclusion.
    """
    def __init__(self, ring, term, ordinal, result):
        self.ring = ring
        self.term = term
        self.ordinal = ordinal
        self.result = weakref.ref(result)

    def process(self, message):
        ring = self.ring
        term = self.term
        result = self.result()
        factors = [result]
        if message is not None:
            message._pyro_dims = result._pyro_dims
            factors.append(message)
        if term._pyro_backward.is_leaf:
            product = ring.sumproduct(factors, set())
            message = ring.broadcast(product, self.ordinal)
        else:
            factors.append(ring.inv(term))
            message = ring.sumproduct(factors, set())
        yield term._pyro_backward, message


class MarginalRing(LogRing):
    """
    Ring of forward-sumproduct backward-marginal operations in log space.
    """
    _backend = 'pyro.ops.einsum.torch_marginal'

    def product(self, term, ordinal):
        result = super().product(term, ordinal)
        if hasattr(term, '_pyro_backward'):
            result._pyro_backward = _MarginalProductBackward(self, term, ordinal, result)
        return result


BACKEND_TO_RING = {
    'torch': LinearRing,
    'pyro.ops.einsum.torch_log': LogRing,
    'pyro.ops.einsum.torch_map': MapRing,
    'pyro.ops.einsum.torch_sample': SampleRing,
    'pyro.ops.einsum.torch_marginal': MarginalRing,
}
