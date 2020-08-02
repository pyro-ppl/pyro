# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import weakref
from abc import ABCMeta, abstractmethod

import torch

from pyro.ops import packed
from pyro.util import jit_iter

SAMPLE_SYMBOL = " "  # must be unique and precede alphanumeric characters


class Backward(object, metaclass=ABCMeta):
    is_leaf = False

    def __call__(self):
        """
        Performs entire backward pass in depth-first order.
        """
        message = None
        stack = [(self, message)]
        while stack:
            bwd, message = stack.pop()
            stack.extend(bwd.process(message))

    @abstractmethod
    def process(self, message):
        raise NotImplementedError


class _LeafBackward(Backward):
    is_leaf = True

    def __init__(self, target):
        self.target = weakref.ref(target)

    def process(self, message):
        target = self.target()
        assert message is not target, 'memory leak'
        target._pyro_backward_result = message
        return ()


def require_backward(tensor):
    """
    Marks a tensor as a leaf in the adjoint graph.
    """
    tensor._pyro_backward = _LeafBackward(tensor)


class _TransposeBackward(Backward):
    def __init__(self, a, axes):
        self.a = a
        self.axes = axes

    def process(self, message):
        if message is None:
            yield self.a._pyro_backward, None
        else:
            inv_axes = [None] * len(self.axes)
            for i, j in enumerate(self.axes):
                inv_axes[j] = i
            yield self.a._pyro_backward, message.permute(inv_axes)


# this requires https://github.com/dgasmith/opt_einsum/pull/74
def transpose(a, axes):
    result = a.permute(axes)
    if hasattr(a, '_pyro_backward'):
        result._pyro_backward = _TransposeBackward(a, axes)
        result._pyro_name = getattr(a, '_pyro_name', '?') + "'"
    return result


def einsum_backward_sample(operands, sample1, sample2):
    """
    Cuts down samples to pass on to subsequent steps.
    This is used in various ``_EinsumBackward.__call__()`` methods.
    This assumes all operands have a ``._pyro_dims`` attribute set.
    """
    # Combine upstream sample with sample at this site.
    if sample1 is None:
        sample = sample2
    elif sample2 is None:
        sample = sample1
    else:
        # Slice sample1 down based on choices in sample2.
        assert set(sample1._pyro_sample_dims).isdisjoint(sample2._pyro_sample_dims)
        sample_dims = sample1._pyro_sample_dims + sample2._pyro_sample_dims
        for dim, index in zip(sample2._pyro_sample_dims, jit_iter(sample2)):
            if dim in sample1._pyro_dims:
                index._pyro_dims = sample2._pyro_dims[1:]
                sample1 = packed.gather(sample1, index, dim)

        # Concatenate the two samples.
        parts = packed.broadcast_all(sample1, sample2)
        sample = torch.cat(parts)
        sample._pyro_dims = parts[0]._pyro_dims
        sample._pyro_sample_dims = sample_dims
        assert sample.dim() == len(sample._pyro_dims)
        if not torch._C._get_tracing_state():
            assert sample.size(0) == len(sample._pyro_sample_dims)

    # Select sample dimensions to pass on to downstream sites.
    for x in operands:
        if not hasattr(x, '_pyro_backward'):
            continue
        if sample is None:
            yield x._pyro_backward, None
            continue
        x_sample_dims = set(x._pyro_dims) & set(sample._pyro_sample_dims)
        if not x_sample_dims:
            yield x._pyro_backward, None
            continue
        if x_sample_dims == set(sample._pyro_sample_dims):
            yield x._pyro_backward, sample
            continue
        x_sample_dims = ''.join(sorted(x_sample_dims))
        x_sample = sample[[sample._pyro_sample_dims.index(dim)
                           for dim in x_sample_dims]]
        x_sample._pyro_dims = sample._pyro_dims
        x_sample._pyro_sample_dims = x_sample_dims
        assert x_sample.dim() == len(x_sample._pyro_dims)
        if not torch._C._get_tracing_state():
            assert x_sample.size(0) == len(x_sample._pyro_sample_dims)
        yield x._pyro_backward, x_sample


def unflatten(flat_sample, output_dims, contract_dims, contract_shape):
    """
    Unpack a collection of indices that have been packed into a 64-bit
    tensor, via modular arithmetic.
    """
    assert contract_dims
    sample = flat_sample.unsqueeze(0)
    if len(contract_dims) > 1:
        slices = [None] * len(contract_dims)
        for i, size in reversed(list(enumerate(contract_shape))):
            slices[i] = sample % size
            sample = sample // size
        sample = torch.cat(slices)
    sample._pyro_dims = SAMPLE_SYMBOL + output_dims
    sample._pyro_sample_dims = contract_dims
    assert sample.dim() == len(sample._pyro_dims)
    if not torch._C._get_tracing_state():
        assert sample.size(0) == len(sample._pyro_sample_dims)
    return sample
