import torch

from pyro.optim import PyroOptim

__all__ = []
# Programmatically load all optimizers from PyTorch.
for _name, _Optim in torch.optim.__dict__.items():
    if not isinstance(_Optim, type):
        continue
    if not issubclass(_Optim, torch.optim.Optimizer):
        continue
    if _Optim is torch.optim.Optimizer:
        continue

    _PyroOptim = (lambda _Optim: lambda optim_args: PyroOptim(_Optim, optim_args))(_Optim)
    _PyroOptim.__name__ = _name
    _PyroOptim.__doc__ = 'Wraps :class:`torch.optim.{}` with :class:`~pyro.optim.optim.PyroOptim`.'.format(_name)

    locals()[_name] = _PyroOptim
    __all__.append(_name)
    del _PyroOptim
