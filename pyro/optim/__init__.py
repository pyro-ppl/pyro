from __future__ import absolute_import, division, print_function

import torch

from .clipped_adam import ClippedAdam as pt_ClippedAdam
from .adagrad_rmsprop import AdagradRMSProp as pt_AdagradRMSProp
from .optim import PyroOptim


def Adam(optim_args):
    """
    A wrapper for :class:`torch.optim.Adam`.
    """
    return PyroOptim(torch.optim.Adam, optim_args)


def Adadelta(optim_args):
    """
    A wrapper for :class:`torch.optim.Adadelta`.
    """
    return PyroOptim(torch.optim.Adadelta, optim_args)


def Adagrad(optim_args):
    """
    A wrapper for :class:`torch.optim.Adagrad`.
    """
    return PyroOptim(torch.optim.Adagrad, optim_args)


def AdagradRMSProp(optim_args):
    """
    A wrapper for an optimizer that is a mash-up of
    :class:`~torch.optim.Adagrad` and :class:`~torch.optim.RMSProp`.
    """
    return PyroOptim(pt_AdagradRMSProp, optim_args)


def Adamax(optim_args):
    """
    A wrapper for :class:`torch.optim.Adamax`.
    """
    return PyroOptim(torch.optim.Adamax, optim_args)


def ASGD(optim_args):
    """
    A wrapper for :class:`torch.optim.ASGD`.
    """
    return PyroOptim(torch.optim.ASGD, optim_args)


def RMSprop(optim_args):
    """
    A wrapper for :class:`torch.optim.RMSprop`.
    """
    return PyroOptim(torch.optim.RMSprop, optim_args)


def Rprop(optim_args):
    """
    A wrapper for :class:`torch.optim.Rprop`.
    """
    return PyroOptim(torch.optim.Rprop, optim_args)


def SGD(optim_args):
    """
    A wrapper for :class:`torch.optim.SGD`.
    """
    return PyroOptim(torch.optim.SGD, optim_args)


def ClippedAdam(optim_args):
    """
    A wrapper for a modification of the :class:`~torch.optim.Adam`
    optimization algorithm that supports gradient clipping.
    """
    return PyroOptim(pt_ClippedAdam, optim_args)
