from __future__ import absolute_import, division, print_function

import weakref

import torch

from pyro.ops import packed

SAMPLE_SYMBOL = " "  # must be unique and precede alphanumeric characters


class _LeafBackward(object):
    def __init__(self, target):
        self.target = weakref.ref(target)

    def __call__(self, result=None):
        target = self.target()
        target._pyro_backward_result = result


def require_backward(tensor):
    """
    Marks a tensor as a leaf in the adjoint graph.
    """
    tensor._pyro_backward = _LeafBackward(tensor)


def requires_backward(tensor):
    """
    Returns true for internal and leaf nodes of the adjoint graph.
    """
    return hasattr(tensor, "_pyro_backward")


class _TransposeBackward(object):
    def __init__(self, a, axes):
        self.a = a
        self.axes = axes

    def __call__(self, sample=None):
        inv_axes = [None] * len(self.axes)
        for i, j in enumerate(self.axes):
            inv_axes[j] = i
        self.a._pyro_backward(sample.permute(inv_axes))


# this requires https://github.com/dgasmith/opt_einsum/pull/74
def transpose(a, axes):
    result = a.permute(axes)
    if requires_backward(a):
        result._pyro_backward = _TransposeBackward(a, axes)
    return result


def einsum_backward_scatter(inputs, operands, sample1, sample2):
    """
    Cut down samples to pass on to subsequent steps.
    This is typically used in ``_EinsumBackward.__call__()`` methods.
    """
    # Combine upstream sample with sample at this site.
    if sample1 is None:
        sample = sample2
    elif sample2 is None:
        sample = sample1
    else:
        sample = sample1
        for dim, index in zip(sample2._pyro_sample_dims, sample2):
            index._pyro_dims = sample2._pyro_dims[1:]
            sample = packed.gather(sample, index, dim)
        parts = packed.broadcast_all(sample, sample2)
        sample = torch.cat(parts)
        sample._pyro_dims = parts[0]._pyro_dims
        sample._pyro_sample_dims = sample1._pyro_sample_dims + sample2._pyro_sample_dims
        assert sample.dim() == len(sample._pyro_dims)
        assert sample.size(0) == len(sample._pyro_sample_dims)

    # Cut down samples to pass on to downstream sites.
    for x_dims, x in zip(inputs, operands):
        if not requires_backward(x):
            continue
        if sample is None:
            x._pyro_backward()
            continue
        needed_dims = set(x_dims) & set(sample._pyro_sample_dims)
        if not needed_dims:
            x._pyro_backward()
            continue
        if needed_dims == set(sample._pyro_sample_dims):
            x._pyro_backward(sample)
            continue
        needed_dims = ''.join(sorted(needed_dims))
        sample_x = sample[[sample._pyro_sample_dims.index(dim)
                           for dim in needed_dims]]
        sample_x._pyro_dims = sample._pyro_dims
        sample_x._pyro_sample_dims = needed_dims
        assert sample_x.dim() == len(sample_x._pyro_dims)
        assert sample_x.size(0) == len(sample_x._pyro_sample_dims)
        x._pyro_backward(sample_x)


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
            sample /= size
        sample = torch.cat(slices)
    sample._pyro_dims = SAMPLE_SYMBOL + output_dims
    sample._pyro_sample_dims = contract_dims
    assert sample.dim() == len(sample._pyro_dims)
    assert sample.size(0) == len(sample._pyro_sample_dims)
    return sample
