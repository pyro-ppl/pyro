# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch


class ProvenanceTensor(torch.Tensor):
    """
    Provenance tracking implementation in Pytorch.

    Provenance of the output tensor is the union of provenances of input tensors.
    """

    def __new__(cls, data, provenance=frozenset(), **kwargs):
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
    provenance = getattr(tensor, "_provenance", frozenset())
    assert isinstance(provenance, frozenset)
    return provenance
