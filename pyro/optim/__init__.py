from .clipped_adam import ClippedAdam
import torch
from .optim import PyroOptim


def Adam(optim_args):
    return PyroOptim(torch.optim.Adam, optim_args)
