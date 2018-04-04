from __future__ import absolute_import, division, print_function

import torch

from .clipped_adam import ClippedAdam as pt_ClippedAdam
from .adagrad_rmsprop import AdagradRMSProp as pt_AdagradRMSProp
from .optim import PyroOptim


def AdagradRMSProp(optim_args):
    """
    A wrapper for an optimizer that is a mash-up of
    :class:`~torch.optim.Adagrad` and :class:`~torch.optim.RMSprop`.
    """
    return PyroOptim(pt_AdagradRMSProp, optim_args)


def ClippedAdam(optim_args):
    """
    A wrapper for a modification of the :class:`~torch.optim.Adam`
    optimization algorithm that supports gradient clipping.
    """
    return PyroOptim(pt_ClippedAdam, optim_args)


# Programmatically load all optimizers from PyTorch.
for _name, _Optim in torch.optim.__dict__.items():
    if not isinstance(_Optim, type):
        continue
    if not issubclass(_Optim, torch.optim.Optimizer):
        continue
    if _Optim is torch.optim.Optimizer:
        continue

    def _PyroOptim(optim_args):
        return PyroOptim(_Optim, optim_args)

    _PyroOptim.__name__ = _name
    _PyroOptim.__doc__ = 'Wraps :class:`torch.optim.{}` with :class:`~pyro.optim.optim.PyroOptim`.'.format(
        _name, _Optim.__name__)

    locals()[_name] = _PyroOptim
    del _PyroOptim
