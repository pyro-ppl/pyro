# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
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
        try:
            functools.update_wrapper(new_fn, old_fn)
        except Exception:
            for attr in functools.WRAPPER_ASSIGNMENTS:
                if hasattr(old_fn, attr):
                    setattr(new_fn, attr, getattr(old_fn, attr))
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


# Fixes a shape error in Multinomial.support with inhomogeneous .total_count
@patch_dependency('torch.distributions.Multinomial.support')
@torch.distributions.constraints.dependent_property
def _Multinomial_support(self):
    total_count = self.total_count
    if isinstance(total_count, torch.Tensor):
        total_count = total_count.unsqueeze(-1)
    return torch.distributions.constraints.integer_interval(0, total_count)


__all__ = []
