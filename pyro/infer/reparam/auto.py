# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import pyro.distributions as dist
from pyro.poutine.reparam_messenger import register_reparam_strategy

from .circular import CircularReparam
from .loc_scale import LocScaleReparam
from .projected_normal import ProjectedNormalReparam
from .reparam import Reparam
from .stable import StableReparam
from .transform import TransformReparam


def _get_base_dist(fn):
    return getattr(fn, "base_dist", fn)


def _can_apply_loc_scale(fn):
    if not {"loc", "scale"}.issubset(fn.arg_constraints):
        return False
    for name in fn.arg_constraints:
        if getattr(fn, name).requires_grad:
            return False
    return True


@register_reparam_strategy("minimal")
def minimal_reparam(msg: dict) -> Optional[Reparam]:
    """
    Minimal reparametrization strategy that reparametrizes only those sites
    that would otherwise lead to error.
    """
    # TODO wrap result
    base_dist = _get_base_dist(msg["fn"])

    if isinstance(base_dist, dist.Stable):
        # TODO
        return StableReparam()

    if msg["value"] is not None:
        return None

    if isinstance(base_dist, dist.ProjectedNormal):
        return ProjectedNormalReparam()

    if isinstance(base_dist, torch.distributions.VonMises):
        return CircularReparam()


@register_reparam_strategy("auto")
def auto_reparam(msg: dict) -> Optional[Reparam]:
    """

    """
    # Apply necessary reparameterizations.
    result = minimal_reparam(msg)
    if result is not None:
        return result

    if msg["value"] is not None:
        return None

    if _can_apply_loc_scale(msg["fn"]):
        return LocScaleReparam()

    # Apply learnable LocScaleRepram wherever possible.
    raise NotImplementedError("TODO")


@register_reparam_strategy("full")
def full_reparam(msg: dict) -> Optional[Reparam]:
    """
    Attempts to fully reparametrize a model such that all remaining samples
    msgs are parameter-free.

    :raises ValueError: In case no reparametrization strategy is available.
    """
    fn = msg["fn"]  # TODO unwrap MaskedDistribution, Independent, Expanded, etc.
    params = set(fn.arg_constraints)

    # If distribution is parameter-free, do nothing.
    if all(not getattr(fn, name).requires_grad for name in params):
        return None

    # Unwrap transforms.
    if isinstance(fn, torch.distributions.TransformedDistribution):
        return TransformReparam()

    # Try to fully decenter the distribution.
    if params.issuperset({"loc", "scale"}):
        for name in params - {"loc", "scale"}:
            if getattr(fn, name).requires_grad:
                raise NotImplementedError
        return LocScaleReparam(0)
