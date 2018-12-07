from __future__ import absolute_import, division, print_function

import os

import opt_einsum
import torch

if 'READTHEDOCS' not in os.environ:
    # RTD is running 0.4.1 due to a memory issue with pytorch 1.0
    assert torch.__version__.startswith('1.')


def patch_dependency(target, root_module=torch):
    parts = target.split('.')
    assert parts[0] == root_module.__name__
    module = root_module
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
def _einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # the old interface of passing the operands as one list argument
        operands = operands[0]

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

    return _einsum._pyro_unpatched(equation, *operands)


# This can be removed after https://github.com/dgasmith/opt_einsum/pull/77 is released.
@patch_dependency('opt_einsum.helpers.compute_size_by_dict', opt_einsum)
def _compute_size_by_dict(indices, idx_dict):
    if torch._C._get_tracing_state():
        # If running under the jit, convert all sizes from tensors to ints, the
        # first time each idx_dict is seen.
        last_idx_dict = getattr(_compute_size_by_dict, '_last_idx_dict', None)
        if idx_dict is not last_idx_dict:
            _compute_size_by_dict._last_idx_dict = idx_dict
            for key, value in idx_dict.items():
                idx_dict[key] = int(value)

    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


__all__ = []
