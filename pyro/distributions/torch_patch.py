from __future__ import absolute_import, division, print_function

import weakref

import torch

assert torch.__version__.startswith('1.')


def patch_dependency(target, root_module=torch):
    parts = target.split('.')
    assert parts[0] == root_module.__name__
    module = root_module
    for part in parts[1:-1]:
        module = getattr(module, part)
    name = parts[-1]
    old_fn = getattr(module, name, None)
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


# TODO: Move upstream to allow for pickle serialization of transforms
@patch_dependency('torch.distributions.transforms.Transform.__getstate__')
def _Transform__getstate__(self):
    attrs = {}
    for k, v in self.__dict__.items():
        if isinstance(v, weakref.ref):
            attrs[k] = None
        else:
            attrs[k] = v
    return attrs


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


# Fixes a shape error in Multinomial.support with inhomogeneous .total_count
@patch_dependency('torch.distributions.Multinomial.support')
@torch.distributions.constraints.dependent_property
def _Multinomial_support(self):
    total_count = self.total_count
    if isinstance(total_count, torch.Tensor):
        total_count = total_count.unsqueeze(-1)
    return torch.distributions.constraints.integer_interval(0, total_count)


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


__all__ = []
