# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
These reparametrization strategies are registered with
:func:`~pyro.poutine.reparam_messenger.register_reparam_strategy` and are
accessed by name via ``poutine.reparam(config=name_of_strategy)`` .
See :func:`~pyro.poutine.handlers.reparam` for usage.
"""

from typing import Optional

import torch

import pyro.distributions as dist
from pyro.poutine.reparam_messenger import register_reparam_strategy

from .loc_scale import LocScaleReparam
from .projected_normal import ProjectedNormalReparam
from .reparam import Reparam
from .softmax import GumbelSoftmaxReparam
from .stable import StableReparam
from .transform import TransformReparam


def _get_base_dist(fn):
    return getattr(fn, "base_dist", fn)


def _can_apply_loc_scale(fn):
    if not {"loc", "scale"}.issubset(fn.arg_constraints):
        return False
    shape_params = sorted(set(fn.arg_constraints) - {"loc", "scale"})

    for name in fn.arg_constraints:
        if getattr(fn, name).requires_grad:
            return False
    return True


def _minimal_reparam(fn, is_observed):
    # Unwrap Independent, Masked, Transformed etc.
    while hasattr(fn, "base_dist"):
        if isinstance(fn, torch.distributions.TransformedDistribution):
            if _minimal_reparam(fn.base_dist) is None:
                return None  # No need to reparametrize.
            else:
                return TransformReparam()  # Then reparametrize new sites.
        fn = fn.base_dist

    if isinstance(fn, dist.Stable):
        return StableReparam()

    if not is_observed:
        return None

    if isinstance(fn, dist.ProjectedNormal):
        return ProjectedNormalReparam()

    # TODO apply CircularReparam for VonMises


@register_reparam_strategy("minimal")
def minimal_reparam(msg: dict) -> Optional[Reparam]:
    """
    Minimal reparametrization strategy that reparametrizes only those sites
    that would otherwise lead to error, e.g.
    :class:`~pyro.distributions.Stable` and
    :class:`~pyro.distributions.ProjectedNormal` random variables.

    Example::

        @poutine.reparam(config="minimal")
        def model(...):
            ...
    """
    return _minimal_reparam(msg["fn"], msg["is_observed"])


@register_reparam_strategy("auto")
def auto_reparam(msg: dict) -> Optional[Reparam]:
    """
    Applies a recommended set of reparametrizers. These currently include:
    :func:`minimal_reparam`,
    :class:`~pyro.infer.reparam.transform.TransformReparam`, a fully-learnable
    :class:`~pyro.infer.reparam.loc_scale.LocScaleReparam`, and
    :class:`~pyro.infer.reparam.softmax.GumbelSoftmaxReparam`.

    Example::

        @poutine.reparam(config="auto")
        def model(...):
            ...
    """
    # Apply minimal reparametrizers.
    result = minimal_reparam(msg)
    if result is not None:
        return result

    # Ignore likelihoods.
    if msg["is_observed"]:
        return None

    # Unwrap Independent, Masked, Transformed etc.
    fn = msg["fn"]
    while hasattr(fn, "base_dist"):
        if isinstance(fn, torch.distributions.TransformedDistribution):
            return TransformReparam()  # Then reparametrize new sites.
        fn = fn.base_dist

    # Apply a learnable LocScaleReparam.
    if {"loc", "scale"}.issubset(fn.arg_constraints):
        shape_params = sorted(set(fn.arg_constraints) - {"loc", "scale"})
        return LocScaleReparam(shape_params=shape_params)

    # Apply SoftmaxReparam.
    if isinstance(fn, torch.distributions.RelaxedOneHotCategorical):
        return GumbelSoftmaxReparam()
