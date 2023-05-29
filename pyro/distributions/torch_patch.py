# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import warnings
import weakref

import torch


def patch_dependency(target, root_module=torch):
    try:
        parts = target.split(".")
        assert parts[0] == root_module.__name__
        module = root_module
        for part in parts[1:-1]:
            module = getattr(module, part)
        name = parts[-1]
        old_fn = getattr(module, name, None)
        # Ensure patching is idempotent.
        old_fn = getattr(old_fn, "_pyro_unpatched", old_fn)

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

    except AttributeError:
        warnings.warn(f"pyro patch_dependency target is stale: {target}")

        def decorator(new_fn):
            return new_fn

    return decorator


# TODO: Move upstream to allow for pickle serialization of transforms
@patch_dependency("torch.distributions.transforms.Transform.__getstate__")
def _Transform__getstate__(self):
    super_ = super(torch.distributions.transforms.Transform, self)
    state = getattr(super_, "__getstate__", self.__dict__.copy)()
    for k, v in state.items():
        if isinstance(v, weakref.ref):
            state[k] = None
    return state


# TODO move upstream
@patch_dependency("torch.distributions.transforms.Transform.clear_cache")
def _Transform_clear_cache(self):
    if self._cache_size == 1:
        self._cached_x_y = None, None


# TODO move upstream
@patch_dependency("torch.distributions.TransformedDistribution.clear_cache")
def _TransformedDistribution_clear_cache(self):
    for t in self.transforms:
        t.clear_cache()


# TODO fix https://github.com/pytorch/pytorch/issues/48054 upstream
@patch_dependency("torch.distributions.HalfCauchy.log_prob")
def _HalfCauchy_logprob(self, value):
    if self._validate_args:
        self._validate_sample(value)
    value = torch.as_tensor(
        value, dtype=self.base_dist.scale.dtype, device=self.base_dist.scale.device
    )
    log_prob = self.base_dist.log_prob(value) + math.log(2)
    log_prob.masked_fill_(value.expand(log_prob.shape) < 0, -float("inf"))
    return log_prob


@patch_dependency("torch.distributions.constraints._CorrCholesky.check")
def _CorrCholesky_check(self, value):
    row_norm = torch.linalg.norm(value.detach(), dim=-1)
    unit_row_norm = (row_norm - 1.0).abs().le(1e-4).all(dim=-1)
    return torch.distributions.constraints.lower_cholesky.check(value) & unit_row_norm


# This adds a __call__ method to satisfy sphinx.
@patch_dependency("torch.distributions.utils.lazy_property.__call__")
def _lazy_property__call__(self):
    raise NotImplementedError


__all__ = []
