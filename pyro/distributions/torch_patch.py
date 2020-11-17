# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
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


# TODO move upstream
@patch_dependency('torch.distributions.transforms.Transform.clear_cache')
def _Transform_clear_cache(self):
    if self._cache_size == 1:
        self._cached_x_y = None, None


# TODO move upstream
@patch_dependency('torch.distributions.TransformedDistribution.clear_cache')
def _TransformedDistribution_clear_cache(self):
    for t in self.transforms:
        t.clear_cache()


# Fixes a shape error in Multinomial.support with inhomogeneous .total_count
@patch_dependency('torch.distributions.Multinomial.support')
@torch.distributions.constraints.dependent_property
def _Multinomial_support(self):
    total_count = self.total_count
    if isinstance(total_count, torch.Tensor):
        total_count = total_count.unsqueeze(-1)
    return torch.distributions.constraints.integer_interval(0, total_count)


# TODO fix https://github.com/pytorch/pytorch/issues/48054 upstream
@patch_dependency('torch.distributions.HalfCauchy.log_prob')
def _HalfCauchy_logprob(self, value):
    value = torch.as_tensor(value, dtype=self.base_dist.scale.dtype,
                            device=self.base_dist.scale.device)
    log_prob = self.base_dist.log_prob(value) + math.log(2)
    log_prob.masked_fill_(value.expand(log_prob.shape) < 0, -float("inf"))
    return log_prob


# This adds a __call__ method to satisfy sphinx.
@patch_dependency('torch.distributions.utils.lazy_property.__call__')
def _lazy_property__call__(self):
    raise NotImplementedError


__all__ = []
