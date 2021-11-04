# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

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

    :param torch.Tensor data: An initial tensor to start tracking.
    :param frozenset provenance: An initial provenance set.
    """

    def __new__(cls, data: torch.Tensor, provenance=frozenset(), **kwargs):
        if not provenance:
            return data
        instance = torch.Tensor.__new__(cls)
        instance.__init__(data, provenance)
        return instance

    def __init__(self, data, provenance=frozenset()):
        assert isinstance(provenance, frozenset)
        if isinstance(data, ProvenanceTensor):
            provenance |= data._provenance
            data = data._t
        self._t = data
        self._provenance = provenance

    def __repr__(self):
        return "Provenance:\n{}\nTensor:\n{}".format(self._provenance, self._t)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # collect provenance information from args
        provenance = frozenset()
        # extract ProvenanceTensor._t data from args
        _args = []
        for arg in args:
            if isinstance(arg, ProvenanceTensor):
                provenance |= arg._provenance
                _args.append(arg._t)
            elif isinstance(arg, (tuple, list)):
                _arg = []
                for a in arg:
                    if isinstance(a, ProvenanceTensor):
                        provenance |= a._provenance
                        _arg.append(a._t)
                    else:
                        _arg.append(a)
                _args.append(tuple(_arg))
            else:
                _args.append(arg)
        ret = func(*_args, **kwargs)
        if isinstance(ret, torch.Tensor):
            return ProvenanceTensor(ret, provenance=provenance)
        if isinstance(ret, (tuple, list)):
            _ret = []
            for r in ret:
                if isinstance(r, torch.Tensor):
                    _ret.append(ProvenanceTensor(r, provenance=provenance))
                else:
                    _ret.append(r)
            return type(ret)(_ret)
        return ret


def get_provenance(tensor: torch.Tensor) -> frozenset:
    """
    Reads the provenance of either a :class:`torch.Tensor` (which may or may
    not be a :class:`ProvenanceTensor` ).

    :param torch.Tensor tensor: An input tensor.
    :returns: A provenance frozenset.
    :rtype: frozenset
    """
    provenance = getattr(tensor, "_provenance", frozenset())
    assert isinstance(provenance, frozenset)
    return provenance


def detach_provenance(tensor: torch.Tensor) -> torch.Tensor:
    """
    Blocks provenance tracking through a tensor, similar to :meth:`torch.Tensor.detach`.

    :param torch.Tensor tensor: An input tensor.
    :returns: A tensor sharing the same data but with no provenance.
    :rtype: torch.Tensor
    """
    while isinstance(tensor, ProvenanceTensor):
        tensor = tensor._t
    assert isinstance(tensor, torch.Tensor)
    return tensor
