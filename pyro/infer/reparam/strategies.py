# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
These reparametrization strategies are registered with
:func:`~pyro.poutine.reparam_messenger.register_reparam_strategy` and are
accessed by name via ``poutine.reparam(config=name_of_strategy)`` .
See :func:`~pyro.poutine.handlers.reparam` for usage.
"""

from abc import abstractmethod
from typing import Callable, Dict, Optional

import torch
from torch.distributions import constraints

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.inspect import get_dependencies
from pyro.poutine.reparam_messenger import BaseStrategy

from .loc_scale import LocScaleReparam
from .projected_normal import ProjectedNormalReparam
from .reparam import Reparam
from .softmax import GumbelSoftmaxReparam
from .stable import LatentStableReparam, StableReparam, SymmetricStableReparam
from .transform import TransformReparam


class Strategy(BaseStrategy):
    """
    Abstract base class for reparametrizer configuration strategies.

    Derived classes must implement the :meth:`configure` method.

    :ivar dict config: A dictionary configuration. This will be populated the
        first time the model is run. Thereafter it can be used as an argument
        to ``poutine.reparam(config=___)``.
    """

    # TODO(https://github.com/pyro-ppl/pyro/issues/2831) As part of refactoring
    # Reparam objects to be nn.Modules:
    # - make Strategy inherit from torch.nn.Module
    # - make self.config a torch.nn.ModuleDict

    def __init__(self):
        # TODO(#2831) Make this a torch.nn.ModuleDict.
        self.config: Dict[str, Optional[Reparam]] = {}
        super().__init__()

    @abstractmethod
    def configure(self, msg: dict) -> Optional[Reparam]:
        """
        Inputs a sample site and returns either None or a
        :class:`~pyro.infer.reparam.reparam.Reparam` instance.

        This will be called only on the first model execution; subsequent
        executions will use the reparametrizer stored in ``self.config``.

        :param dict msg: A sample site to possibly reparametrize.
        :returns: An optional reparametrizer instance.
        """
        raise NotImplementedError

    def __call__(self, fn: Callable):
        """
        Strategies can be used as decorators to reparametrize a model.

        :param msg_or_fn: Public use: a model to be decorated.
        """
        return poutine.reparam(fn, self)

    def config_with_model(
        self,
        msg: dict,
        model: Callable,
        model_args: tuple,
        model_kwargs: dict,
    ) -> Optional[Reparam]:
        name = msg["name"]
        if name in self.config:
            return self.config[name]
        result = self.configure(msg)
        self.config[name] = result
        return result


class MinimalReparam(Strategy):
    """
    Minimal reparametrization strategy that reparametrizes only those sites
    that would otherwise lead to error, e.g.
    :class:`~pyro.distributions.Stable` and
    :class:`~pyro.distributions.ProjectedNormal` random variables.

    Example::

        @MinimalReparam()
        def model(...):
            ...

    which is equivalent to::

        @poutine.reparam(config=MinimalReparam())
        def model(...):
            ...
    """

    def configure(self, msg: dict) -> Optional[Reparam]:
        return _minimal_reparam(msg["fn"], msg["is_observed"])


def _minimal_reparam(fn, is_observed):
    # Unwrap Independent, Masked, Transformed etc.
    while isinstance(getattr(fn, "base_dist", None), dist.Distribution):
        if isinstance(fn, torch.distributions.TransformedDistribution):
            if _minimal_reparam(fn.base_dist, is_observed) is None:
                return None  # No need to reparametrize.
            else:
                return TransformReparam()  # Then reparametrize new sites.
        fn = fn.base_dist

    if isinstance(fn, dist.Stable):
        if not is_observed:
            return LatentStableReparam()
        elif fn.skew.requires_grad or fn.skew.any():
            return StableReparam()
        else:
            return SymmetricStableReparam()

    if isinstance(fn, dist.ProjectedNormal):
        return ProjectedNormalReparam()

    # TODO apply CircularReparam for VonMises


class AutoReparam(Strategy):
    """
    Applies a recommended set of reparametrizers. These currently include:
    :class:`MinimalReparam`,
    :class:`~pyro.infer.reparam.transform.TransformReparam`, a fully-learnable
    :class:`~pyro.infer.reparam.loc_scale.LocScaleReparam`, and
    :class:`~pyro.infer.reparam.softmax.GumbelSoftmaxReparam`.

    Example::

        @AutoReparam()
        def model(...):
            ...

    which is equivalent to::

        @poutine.reparam(config=AutoReparam())
        def model(...):
            ...

    .. warning:: This strategy may change behavior across Pyro releases.
        To inspect or save a given behavior, extract the ``.config`` dict after
        running the model at least once.

    :param centered: Optional centering parameter for
        :class:`~pyro.infer.reparam.loc_scale.LocScaleReparam` reparametrizers.
        If None (default), centering will be learned. If a float in
        ``[0.0,1.0]``, then a fixed centering. To completely decenter (e.g. in
        MCMC), set to 0.0.
    """

    def __init__(self, *, centered: Optional[float] = None):
        assert centered is None or isinstance(centered, float)
        super().__init__()
        self.centered = centered
        self.dependencies = None

    def config_with_model(
        self,
        msg: dict,
        model: Callable,
        model_args: tuple,
        model_kwargs: dict,
    ) -> Optional[Reparam]:
        if self.dependencies is None:
            self.dependencies = get_dependencies(model, model_args, model_kwargs)
        super().config_with_model(self, msg, model, model_args, model_kwargs)

    def configure(self, msg: dict) -> Optional[Reparam]:
        # Focus on tricks for latent sites.
        fn = msg["fn"]
        if not msg["is_observed"]:
            # Unwrap Independent, Masked, Transformed etc.
            while isinstance(getattr(fn, "base_dist", None), dist.Distribution):
                if isinstance(fn, torch.distributions.TransformedDistribution):
                    return TransformReparam()  # Then reparametrize new sites.
                fn = fn.base_dist

            # Try to apply a GumbelSoftmaxReparam.
            if isinstance(fn, torch.distributions.RelaxedOneHotCategorical):
                return GumbelSoftmaxReparam()

            # Check whether parameters depend on upstream latent variables.
            if len(self.dependencies["prior_dependencies"][msg["name"]]) > 1:
                # Apply a learnable LocScaleReparam.
                result = _loc_scale_reparam(msg["name"], fn, self.centered)
                if result is not None:
                    return result

        # Apply minimal reparametrizers.
        return _minimal_reparam(fn, msg["is_observed"])


def _loc_scale_reparam(name, fn, centered):
    if "_decentered" in name:
        return  # Avoid infinite recursion.

    # Check for location-scale families.
    params = set(fn.arg_constraints)
    if not {"loc", "scale"}.issubset(params):
        return

    # Check for unconstrained support.
    if not _is_unconstrained(fn.support):
        return

    # Create an elementwise-learnable reparametrizer.
    shape_params = sorted(params - {"loc", "scale"})
    return LocScaleReparam(centered=centered, shape_params=shape_params)


def _is_unconstrained(constraint):
    # Unwrap constraints.independent.
    while hasattr(constraint, "base_constraint"):
        constraint = constraint.base_constraint
    return constraint == constraints.real
