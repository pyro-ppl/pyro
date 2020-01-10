# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.optim import PyroOptim
from pyro.optim.lr_scheduler import PyroLRScheduler

__all__ = []
# Programmatically load all optimizers from PyTorch.
for _name, _Optim in torch.optim.__dict__.items():
    if not isinstance(_Optim, type):
        continue
    if not issubclass(_Optim, torch.optim.Optimizer):
        continue
    if _Optim is torch.optim.Optimizer:
        continue
    if _Optim is torch.optim.LBFGS:
        # XXX LBFGS is not supported for SVI yet
        continue

    _PyroOptim = (lambda _Optim: lambda optim_args, clip_args=None: PyroOptim(_Optim, optim_args, clip_args))(_Optim)
    _PyroOptim.__name__ = _name
    _PyroOptim.__doc__ = 'Wraps :class:`torch.optim.{}` with :class:`~pyro.optim.optim.PyroOptim`.'.format(_name)

    locals()[_name] = _PyroOptim
    __all__.append(_name)
    del _PyroOptim

# Load all schedulers from PyTorch
for _name, _Optim in torch.optim.lr_scheduler.__dict__.items():
    if not isinstance(_Optim, type):
        continue
    if not issubclass(_Optim, torch.optim.lr_scheduler._LRScheduler) and _name != 'ReduceLROnPlateau':
        continue
    if _Optim is torch.optim.Optimizer:
        continue

    _PyroOptim = (
        lambda _Optim: lambda optim_args, clip_args=None: PyroLRScheduler(_Optim, optim_args, clip_args)
    )(_Optim)
    _PyroOptim.__name__ = _name
    _PyroOptim.__doc__ = 'Wraps :class:`torch.optim.{}` with '.format(_name) +\
                         ':class:`~pyro.optim.lr_scheduler.PyroLRScheduler`.'

    locals()[_name] = _PyroOptim
    __all__.append(_name)
    del _PyroOptim
