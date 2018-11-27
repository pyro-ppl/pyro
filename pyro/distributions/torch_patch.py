from __future__ import absolute_import, division, print_function

import torch

from pyro.util import patch_dependency


@patch_dependency('torch._standard_gamma')
def _torch_standard_gamma(concentration):
    unpatched_fn = _torch_standard_gamma._pyro_unpatched
    if concentration.is_cuda:
        return unpatched_fn(concentration.cpu()).cuda(concentration.get_device())
    return unpatched_fn(concentration)


@patch_dependency('torch.distributions.gamma._standard_gamma')
def _standard_gamma(concentration):
    if concentration.is_cuda:
        return concentration.cpu()._standard_gamma().cuda(concentration.get_device())
    return concentration._standard_gamma()


@patch_dependency('torch._dirichlet_grad')
def _torch_dirichlet_grad(x, concentration, total):
    unpatched_fn = _torch_dirichlet_grad._pyro_unpatched
    if x.is_cuda:
        return unpatched_fn(x.cpu(), concentration.cpu(), total.cpu()).cuda(x.get_device())
    return unpatched_fn(x, concentration, total)


# This can be removed when super(...).__init__() is added upstream
@patch_dependency('torch.distributions.transforms.Transform.__init__')
def _Transform__init__(self, cache_size=0):
    self._cache_size = cache_size
    self._inv = None
    if cache_size == 0:
        pass  # default behavior
    elif cache_size == 1:
        self._cached_x_y = None, None
    else:
        raise ValueError('cache_size must be 0 or 1')
    super(torch.distributions.transforms.Transform, self).__init__()


@patch_dependency('torch.linspace')
def _torch_linspace(*args, **kwargs):
    unpatched_fn = _torch_linspace._pyro_unpatched
    template = torch.Tensor()
    if template.is_cuda:
        kwargs["device"] = "cpu"
        ret = unpatched_fn(*args, **kwargs).to(device=template.device)
        kwargs.pop("device", None)
    else:
        ret = unpatched_fn(*args, **kwargs)
    return ret


@patch_dependency('torch.einsum')
def _einsum(equation, operands):
    # work around torch.einsum performance issues
    # see https://github.com/pytorch/pytorch/issues/10661
    if equation == 'ac,abc->bc':
        x, y = operands
        return (x.unsqueeze(1) * y).sum(0)
    elif equation == 'ac,abc->cb':
        x, y = operands
        return (x.unsqueeze(1) * y).sum(0).transpose(0, 1)
    elif equation == 'abc,ac->cb':
        y, x = operands
        return (x.unsqueeze(1) * y).sum(0).transpose(0, 1)

    # this workaround can be deleted after this issue is fixed in release:
    # https://github.com/pytorch/pytorch/issues/7763
    operands = [t.clone() for t in operands]

    return _einsum._pyro_unpatched(equation, operands)


__all__ = []
