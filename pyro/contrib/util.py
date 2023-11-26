# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import weakref
from collections import OrderedDict

import torch

import pyro


def get_indices(labels, sizes=None, tensors=None):
    indices = []
    start = 0
    if sizes is None:
        sizes = OrderedDict([(l, t.shape[0]) for l, t in tensors.items()])
    for label in sizes:
        end = start + sizes[label]
        if label in labels:
            indices.extend(range(start, end))
        start = end
    return torch.tensor(indices)


def tensor_to_dict(sizes, tensor, subset=None):
    if subset is None:
        subset = sizes.keys()
    start = 0
    out = {}
    for label, size in sizes.items():
        end = start + size
        if label in subset:
            out[label] = tensor[..., start:end]
        start = end
    return out


def rmm(A, B):
    """Shorthand for `matmul`."""
    return torch.matmul(A, B)


def rmv(A, b):
    """Tensorized matrix vector multiplication of rightmost dimensions."""
    return torch.matmul(A, b.unsqueeze(-1)).squeeze(-1)


def rvv(a, b):
    """Tensorized vector vector multiplication of rightmost dimensions."""
    return torch.matmul(a.unsqueeze(-2), b.unsqueeze(-1)).squeeze(-2).squeeze(-1)


def lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    return A.expand(tuple(dimensions) + A.shape)


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,) * len(dimensions)).expand(A.shape + tuple(dimensions))


def rdiag(v):
    """Converts the rightmost dimension to a diagonal matrix."""
    return rexpand(v, v.shape[-1]) * torch.eye(v.shape[-1])


def rtril(M, diagonal=0, upper=False):
    """Takes the lower-triangular of the rightmost 2 dimensions."""
    if upper:
        return rtril(M, diagonal=diagonal, upper=False).transpose(-1, -2)
    return M * torch.tril(torch.ones(M.shape[-2], M.shape[-1]), diagonal=diagonal)


def iter_plates_to_shape(shape):
    # Go backwards (right to left)
    for i, s in enumerate(shape[::-1]):
        yield pyro.plate("plate_" + str(i), s)


def check_no_weakref(obj, path="", avoid_ids=None):
    """Attempts to check that an object has no weakrefs."""
    if avoid_ids is None:
        avoid_ids = {id(obj)}

    if isinstance(obj, weakref.ref):
        raise ValueError(f"Weakref found at {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if id(v) not in avoid_ids:
                check_no_weakref(v, path + f"[{k}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if id(v) not in avoid_ids:
                check_no_weakref(v, path + f"[{i}]")
    elif hasattr(obj, "__dict__"):
        check_no_weakref(obj.__dict__, path)
