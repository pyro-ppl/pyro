import torch
from pyro.optim import PyroOptim
from torchcontrib.optim import SWA as _SWA

def _swa_constructor(
    param: torch.nn.Parameter, 
    base: type, 
    base_args: dict, 
    swa_args: dict,
) -> torch.optim.Optimizer:
    base = base(param, **base_args)
    optimizer = _SWA(base, **swa_args)
    return optimizer

def SWA(args: dict) -> PyroOptim:
    """ 
    Stochastic Weight Averaging (SWA) optimizer. [1]

    References:
    [1] 'Averaging Weights Leads to Wider Optima and Better Generalization', 
    Pavel Izmailov, Dmitry Podoprikhin, Timur Garipov, Dmitry Vetrov, 
    Andrew Gordon Wilson
    Uncertainty in Artificial Intelligence (UAI), 2018

    Arguments:
    :param args: arguments for SWA optimizer
    
    """
    return PyroOptim(_swa_constructor, args)

def swap_swa_sgd(optimizer: PyroOptim) -> None:
    """
    Swap the SWA optimized parameters with samples.

    Arguments:
    :param optimizer: SWA optimizer
    """
    for key, value in optimizer.optim_objs.items():
        value.swap_swa_sgd()
