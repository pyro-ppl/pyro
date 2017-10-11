from .clipped_adam import ClippedAdam as pt_ClippedAdam
import torch
from .optim import PyroOptim


def Adam(optim_args):
    return PyroOptim(torch.optim.Adam, optim_args)


def ClippedAdam(optim_args):
    return PyroOptim(pt_ClippedAdam, optim_args)
