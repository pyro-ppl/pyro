import torch
import torch.nn.functional as functional
from torch.autograd import Variable


def log_gamma(xx):
    if isinstance(xx, Variable):
        ttype = xx.data.type()
    elif isinstance(xx, torch.Tensor):
        ttype = xx.type()
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = Variable(torch.ones(x.size()).type(ttype)) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def log_beta(t):
    """
    Computes log Beta function.

    :param t:
    :type t: torch.autograd.Variable of dimension 1 or 2
    :rtype: torch.autograd.Variable of float (if t.dim() == 1) or torch.Tensor (if t.dim() == 2)
    """
    assert t.dim() in (1, 2)
    if t.dim() == 1:
        numer = torch.sum(log_gamma(t))
        denom = log_gamma(torch.sum(t))
    else:
        numer = torch.sum(log_gamma(t), 1)
        denom = log_gamma(torch.sum(t, 1))
    return numer - denom


def torch_zeros_like(x):
    """
    Polyfill for `torch.zeros_like()`.
    """
    # Work around https://github.com/pytorch/pytorch/issues/2906
    if isinstance(x, Variable):
        return Variable(torch_zeros_like(x.data))
    # Support Pytorch before https://github.com/pytorch/pytorch/pull/2489
    try:
        return torch.zeros_like(x)
    except AttributeError:
        return torch.zeros(x.size()).type_as(x)


def torch_ones_like(x):
    """
    Polyfill for `torch.ones_like()`.
    """
    # Work around https://github.com/pytorch/pytorch/issues/2906
    if isinstance(x, Variable):
        return Variable(torch_ones_like(x.data))
    # Support Pytorch before https://github.com/pytorch/pytorch/pull/2489
    try:
        return torch.ones_like(x)
    except AttributeError:
        return torch.ones(x.size()).type_as(x)


def softmax(x, dim=-1):
    """
    TODO: change to use the default pyTorch implementation when available
    Source: https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
    :param x: tensor
    :param dim: Dimension to apply the softmax function to. The elements of the tensor in this
        dimension must sum to 1.
    :return: tensor having the same dimension as `x` rescaled along dim
    """
    input_size = x.size()

    trans_input = x.transpose(dim, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = functional.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(dim, len(input_size) - 1)
