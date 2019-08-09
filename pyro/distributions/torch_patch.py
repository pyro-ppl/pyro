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


# This can be removed after release of https://github.com/pytorch/pytorch/pull/24131
@patch_dependency('torch.distributions.LowerCholeskyTransform._call')
def _LowerCholeskyTransform_call(self, x):
    return x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()


# This can be removed after release of https://github.com/pytorch/pytorch/pull/24131
@patch_dependency('torch.distributions.LowerCholeskyTransform._inverse')
def _LowerCholeskyTransform_inverse(self, y):
    return y.tril(-1) + y.diagonal(dim1=-2, dim2=-1).log().diag_embed()


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

    return _einsum._pyro_unpatched(equation, *operands)


__all__ = []
