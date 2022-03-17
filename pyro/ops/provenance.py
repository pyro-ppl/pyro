# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch
from typing import Tuple

import torch


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

    def __new__(cls, data: torch.Tensor, provenance=frozenset(), **kwargs):
        assert not isinstance(data, ProvenanceTensor)
        if not provenance:
            return data
        return super().__new__(cls)

    def __init__(self, data, provenance=frozenset()):
        assert isinstance(provenance, frozenset)
        if isinstance(data, ProvenanceTensor):
            provenance |= data._provenance
            data = data._t
        self._t = data
        self._provenance = provenance

    def __repr__(self):
        return "Provenance:\n{}\nTensor:\n{}".format(self._provenance, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # collect provenance information from args
        provenance = frozenset()
        # extract ProvenanceTensor._t data from args and kwargs
        _args = []
        for arg in args:
            _arg, _provenance = extract_provenance(arg)
            _args.append(_arg)
            provenance |= _provenance
        _kwargs = {}
        for k, v in kwargs.items():
            _v, _provenance = extract_provenance(v)
            _kwargs[k] = _v
            provenance |= provenance
        ret = func(*_args, **_kwargs)
        _ret = track_provenance(ret, provenance)
        return _ret


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
@track_provenance.register(list)
@track_provenance.register(set)
@track_provenance.register(tuple)
def _track_provenance_list(x, provenance: frozenset):
    return type(x)(track_provenance(part, provenance) for part in x)


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
@extract_provenance.register(list)
@extract_provenance.register(set)
@extract_provenance.register(tuple)
def _extract_provenance_list(x):
    provenance = frozenset()
    values = []
    for part in x:
        v, p = extract_provenance(part)
        values.append(v)
        provenance |= p
    value = type(x)(values)
    return value, provenance


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


def detach_provenance(x):
    """
    Blocks provenance tracking through a tensor, similar to :meth:`torch.Tensor.detach`.

    :param torch.Tensor tensor: An input tensor.
    :returns: A tensor sharing the same data but with no provenance.
    :rtype: torch.Tensor
    """
    value, _ = extract_provenance(x)
    return value
