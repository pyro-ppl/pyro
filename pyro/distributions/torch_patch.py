from __future__ import absolute_import, division, print_function

import warnings

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


@_patch('torch._dirichlet_grad')
def _torch_dirichlet_grad(x, concentration, total):
    unpatched_fn = _torch_dirichlet_grad._pyro_unpatched
    if x.is_cuda:
        return unpatched_fn(x.cpu(), concentration.cpu(), total.cpu()).cuda(x.get_device())
    return unpatched_fn(x, concentration, total)


@_patch('torch.distributions.utils._default_promotion')
def _default_promotion(v):
    # Ignore jit warnings about promoting Python numbers to tensors,
    # assuming all numbers are constant literals.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning,
                                message="torch.tensor might cause the trace to be incorrect")
        return _default_promotion._pyro_unpatched(v)


@_patch('torch.einsum')
def _einsum(equation, operands):
    # work around torch.einsum performance issues
    # see https://github.com/pytorch/pytorch/issues/10661
    if equation == 'ac,abc->cb':
        x, y = operands
        return (x.unsqueeze(1) * y).sum(0).transpose(0, 1)
    elif equation == 'abc,ac->cb':
        y, x = operands
        return (x.unsqueeze(1) * y).sum(0).transpose(0, 1)

    return _einsum._pyro_unpatched(equation, operands)


__all__ = []
