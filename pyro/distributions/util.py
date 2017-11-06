from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def log_gamma(xx):
    if isinstance(xx, torch.Tensor):
        xx = Variable(xx)
    ttype = xx.data.type()
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
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


def move_to_same_host_as(source, destin):
    """
    Returns source or a copy of `source` such that `source.is_cuda == `destin.is_cuda`.
    """
    return source.cuda() if destin.is_cuda else source.cpu()


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


def torch_eye(n, m=None, out=None):
    """
    Like `torch.eye()`, but works with cuda tensors.
    """
    if m is None:
        m = n
    try:
        return torch.eye(n, m, out=out)
    except TypeError:
        # Only catch errors due to torch.eye() not being available for cuda tensors.
        module = torch.Tensor.__module__ if out is None else type(out).__module__
        if module != 'torch.cuda':
            raise
    Tensor = getattr(torch, torch.Tensor.__name__)
    cpu_out = Tensor(n, m)
    cuda_out = torch.eye(m, n, out=cpu_out).cuda()
    return cuda_out if out is None else out.copy_(cuda_out)


def torch_multinomial(input, num_samples, replacement=False):
    """
    Like `torch.multinomial()` but works with cuda tensors.
    Does not support keyword argument `out`.
    """
    if input.is_cuda:
        return torch_multinomial(input.cpu(), num_samples, replacement).cuda()
    else:
        return torch.multinomial(input, num_samples, replacement)


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

    try:
        soft_max_2d = F.softmax(input_2d, 1)
    except TypeError:
        # Support older pytorch 0.2 release.
        soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(dim, len(input_size) - 1)


def _get_clamping_buffer(tensor):
    clamp_eps = 1e-6
    if isinstance(tensor, Variable):
        tensor = tensor.data
    if isinstance(tensor, (torch.DoubleTensor, torch.cuda.DoubleTensor)):
        clamp_eps = 1e-15
    return clamp_eps


def get_probs_and_logits(ps=None, logits=None, is_multidimensional=True):
    """
    Convert probability values to logits, or vice-versa. Either ``ps`` or
    ``logits`` should be specified, but not both.

    :param ps: tensor of probabilities. Should be in the interval *[0, 1]*.
        If, ``is_multidimensional = True``, then must be normalized along
        axis -1.
    :param logits: tensor of logit values.  For the multidimensional case,
        the values, when exponentiated along the last dimension, must sum
        to 1.
    :param is_multidimensional: determines the computation of ps from logits,
        and vice-versa. For the multi-dimensional case, logit values are
        assumed to be log probabilities, whereas for the uni-dimensional case,
        it specifically refers to log odds.
    :return: tuple containing raw probabilities and logits as tensors.
    """
    assert (ps is None) != (logits is None)
    if ps is not None:
        eps = _get_clamping_buffer(ps)
        ps_clamped = ps.clamp(min=eps, max=1 - eps)
    if is_multidimensional:
        if ps is None:
            ps = softmax(logits, -1)
        else:
            logits = torch.log(ps_clamped)
    else:
        if ps is None:
            ps = F.sigmoid(logits)
        else:
            logits = torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return ps, logits
