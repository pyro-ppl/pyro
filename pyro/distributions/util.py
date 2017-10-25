import torch
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
        # Only catch errors due to torch.eye() not being availble for cuda tensors.
        module = torch.Tensor.__module__ if out is None else type(out).__module__
        if module != 'torch.cuda':
            raise
    Tensor = getattr(torch, torch.Tensor.__name__)
    cpu_out = Tensor(n, m)
    cuda_out = torch.eye(m, n, out=cpu_out).cuda()
    return cuda_out if out is None else out.copy_(cuda_out)


def torch_multinomial(input, num_samples, replacement=False):
    """
    Like `torch.multinomial()` but with full support for cuda tensors.
    Does not support keyword argument `out`.
    """
    try:
        return torch.multinomial(input, num_samples, replacement)
    except RuntimeError:
        # Only catch errors due to oversized inputs on cuda.
        if input.dim() <= 2:
            raise
    flat_input = input.view(-1, input.size()[-1])
    flat_out = torch.multinomial(flat_input, num_samples, replacement)
    return flat_out.view(input.size())
