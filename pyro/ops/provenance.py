# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial, singledispatch
from typing import Tuple, TypeVar

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

_Tensor = TypeVar("_Tensor", bound=torch.Tensor)


class ProvenanceTensor(torch.Tensor):
    """
    Provenance tracking implementation in Pytorch.

    This class wraps a :class:`torch.Tensor` to track provenance through
    PyTorch ops, where provenance is a user-defined frozenset of objects. The
    provenance of the output tensors of any op is the union of provenances of
    input tensors.

    -   To start tracking provenance, wrap a :class:`torch.Tensor` in a
        :class:`ProvenanceTensor` with user-defined initial provenance.
    -   To read the provenance of a tensor use :meth:`get_provenance` .
    -   To detach provenance during a computation (similar to
        :meth:`~torch.Tensor.detach` to detach gradients during Pytorch
        computations), use the :meth:`detach_provenance` . This is useful to
        distinguish direct vs indirect provenance.

    Example::

        >>> a = ProvenanceTensor(torch.randn(3), frozenset({"a"}))
        >>> b = ProvenanceTensor(torch.randn(3), frozenset({"b"}))
        >>> c = torch.randn(3)
        >>> assert get_provenance(a + b + c) == frozenset({"a", "b"})
        >>> assert get_provenance(a + detach_provenance(b) + c) == frozenset({"a"})

    **References**

    [1] David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind (2011)
        Nonstandard Interpretations of Probabilistic Programs for Efficient Inference
        http://papers.neurips.cc/paper/4309-nonstandard-interpretations-of-probabilistic-programs-for-efficient-inference.pdf

    :param torch.Tensor data: An initial tensor to start tracking.
    :param frozenset provenance: An initial provenance set.
    """

    _t: torch.Tensor
    _provenance: frozenset

    def __new__(cls, data: torch.Tensor, provenance=frozenset(), **kwargs):
        assert not isinstance(data, ProvenanceTensor)
        if not provenance:
            return data
        ret = data.as_subclass(cls)
        ret._t = data  # this makes sure that detach_provenance always
        # returns the same object. This is important when
        # using the tensor as key in a dict, e.g. the global
        # param store
        ret._provenance = provenance
        return ret

    def __repr__(self):
        return "Provenance:\n{}\nTensor:\n{}".format(self._provenance, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        _args, _kwargs = detach_provenance([args, kwargs or {}])
        ret = func(*_args, **_kwargs)
        return track_provenance(ret, get_provenance([args, kwargs]))


@singledispatch
def track_provenance(x, provenance: frozenset):
    """
    Adds provenance info to the :class:`torch.Tensor` leaves of a data structure.

    :param x: an object to add provenence info to.
    :param frozenset provenance: A provenence set.
    :returns: A provenence-tracking version of ``x``.
    """
    return x


track_provenance.register(torch.Tensor)(ProvenanceTensor)


@track_provenance.register(frozenset)
@track_provenance.register(set)
def _track_provenance_set(x, provenance: frozenset):
    return type(x)(track_provenance(part, provenance) for part in x)


@track_provenance.register(list)
@track_provenance.register(tuple)
@track_provenance.register(dict)
def _track_provenance_pytree(x, provenance: frozenset):
    return tree_map(partial(track_provenance, provenance=provenance), x)


@track_provenance.register
def _track_provenance_provenancetensor(x: ProvenanceTensor, provenance: frozenset):
    x_value, old_provenance = extract_provenance(x)
    return track_provenance(x_value, old_provenance | provenance)


@singledispatch
def extract_provenance(x) -> Tuple[object, frozenset]:
    """
    Extracts the provenance of a data structure possibly containing
    :class:`torch.Tensor` s as leaves, and separates into a detached object and
    provenance.

    :param x: An input data structure.
    :returns: a tuple ``(detached_value, provenance)``
    :rtype: tuple
    """
    return x, frozenset()


@extract_provenance.register(ProvenanceTensor)
def _extract_provenance_tensor(x):
    return x._t, x._provenance


@extract_provenance.register(frozenset)
@extract_provenance.register(set)
def _extract_provenance_set(x):
    provenance = frozenset()
    values = []
    for part in x:
        v, p = extract_provenance(part)
        values.append(v)
        provenance |= p
    value = type(x)(values)
    return value, provenance


@extract_provenance.register(list)
@extract_provenance.register(tuple)
@extract_provenance.register(dict)
def _extract_provenance_pytree(x):
    flat_args, spec = tree_flatten(x)
    xs = []
    provenance = frozenset()
    for x, p in map(extract_provenance, flat_args):
        xs.append(x)
        provenance |= p
    return tree_unflatten(xs, spec), provenance


def get_provenance(x) -> frozenset:
    """
    Reads the provenance of a recursive datastructure possibly containing
    :class:`torch.Tensor` s.

    :param torch.Tensor tensor: An input tensor.
    :returns: A provenance frozenset.
    :rtype: frozenset
    """
    _, provenance = extract_provenance(x)
    return provenance


def detach_provenance(x: _Tensor) -> _Tensor:
    """
    Blocks provenance tracking through a tensor, similar to :meth:`torch.Tensor.detach`.

    :param torch.Tensor tensor: An input tensor.
    :returns: A tensor sharing the same data but with no provenance.
    :rtype: torch.Tensor
    """
    value, _ = extract_provenance(x)
    return value  # type: ignore[return-value]
