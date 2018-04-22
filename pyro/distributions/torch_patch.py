from __future__ import absolute_import, division, print_function

import torch


def _patch(target):
    parts = target.split('.')
    assert parts[0] == 'torch'
    module = torch
    for part in parts[1:-1]:
        module = getattr(module, part)
    name = parts[-1]
    old_fn = getattr(module, name)
    old_fn = getattr(old_fn, '_pyro_unpatched', old_fn)  # ensure patching is idempotent

    def decorator(new_fn):
        new_fn.__name__ = name
        new_fn._pyro_unpatched = old_fn
        setattr(module, name, new_fn)
        return new_fn

    return decorator


@_patch('torch._standard_gamma')
def _torch_standard_gamma(concentration):
    unpatched_fn = _torch_standard_gamma._pyro_unpatched
    if concentration.is_cuda:
        return unpatched_fn(concentration.cpu()).cuda(concentration.get_device())
    return unpatched_fn(concentration)


@_patch('torch.distributions.gamma._standard_gamma')
def _standard_gamma(concentration):
    if concentration.is_cuda:
        return concentration.cpu()._standard_gamma().cuda(concentration.get_device())
    return concentration._standard_gamma()


@_patch('torch._dirichlet_grad')
def _torch_dirichlet_grad(x, concentration, total):
    unpatched_fn = _torch_dirichlet_grad._pyro_unpatched
    if x.is_cuda:
        return unpatched_fn(x.cpu(), concentration.cpu(), total.cpu()).cuda(x.get_device())
    return unpatched_fn(x, concentration, total)


__all__ = []
