# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
import torch.nn

from .torch import TransformedDistribution
from .torch_transform import ComposeTransformModule


class ConditionalDistribution(ABC):
    @abstractmethod
    def condition(self, context):
        """:rtype: torch.distributions.Distribution"""
        raise NotImplementedError


class ConditionalTransform(ABC):
    @abstractmethod
    def condition(self, context):
        """:rtype: torch.distributions.Transform"""
        raise NotImplementedError


class ConditionalTransformModule(ConditionalTransform, torch.nn.Module):
    """
    Conditional transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than :class:`~pyro.distributions.conditional.ConditionalTransform` so they are also a subclass of
    :class:`~torch.nn.Module` and inherit all the useful methods of that class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return super().__hash__()

    @property
    def inv(self) -> "ConditionalTransformModule":
        return _ConditionalInverseTransformModule(self)


class _ConditionalInverseTransformModule(ConditionalTransformModule):
    def __init__(self, transform: ConditionalTransform):
        super().__init__()
        self._transform = transform

    @property
    def inv(self) -> ConditionalTransform:
        return self._transform

    def condition(self, context: torch.Tensor):
        return self._transform.condition(context).inv


class ConditionalComposeTransformModule(
    ConditionalTransformModule, torch.nn.ModuleList
):
    """
    Conditional analogue of :class:`~pyro.distributions.torch_transform.ComposeTransformModule` .

    Useful as a base class for specifying complicated conditional distributions::

        >>> class ConditionalFlowStack(dist.conditional.ConditionalComposeTransformModule):
        ...     def __init__(self, input_dim, context_dim, hidden_dims, num_flows):
        ...         super().__init__([
        ...             dist.transforms.conditional_planar(input_dim, context_dim, hidden_dims)
        ...             for _ in range(num_flows)
        ...         ], cache_size=1)

        >>> cond_dist = dist.conditional.ConditionalTransformedDistribution(
        ...     dist.Normal(torch.zeros(3), torch.ones(3)).to_event(1),
        ...     [ConditionalFlowStack(3, 2, [8, 8], num_flows=4).inv]
        ... )

        >>> context = torch.rand(10, 2)
        >>> data = torch.rand(10, 3)
        >>> nll = -cond_dist.condition(context).log_prob(data)
    """

    def __init__(self, transforms, cache_size: int = 0):
        self.transforms = [
            (
                ConstantConditionalTransform(t)
                if not isinstance(t, ConditionalTransform)
                else t
            )
            for t in transforms
        ]
        super().__init__()
        if cache_size not in {0, 1}:
            raise ValueError("cache_size must be 0 or 1")
        self._cache_size = cache_size
        # for parameter storage
        for t in transforms:
            if isinstance(t, torch.nn.Module):
                self.append(t)

    def condition(self, context: torch.Tensor) -> ComposeTransformModule:
        return ComposeTransformModule(
            [t.condition(context) for t in self.transforms]
        ).with_cache(self._cache_size)


class ConstantConditionalDistribution(ConditionalDistribution):
    def __init__(self, base_dist):
        assert isinstance(base_dist, torch.distributions.Distribution)
        self.base_dist = base_dist

    def condition(self, context):
        return self.base_dist


class ConstantConditionalTransform(ConditionalTransform):
    def __init__(self, transform):
        assert isinstance(transform, torch.distributions.Transform)
        self.transform = transform

    def condition(self, context):
        return self.transform

    def clear_cache(self):
        self.transform.clear_cache()


class ConditionalTransformedDistribution(ConditionalDistribution):
    def __init__(self, base_dist, transforms):
        self.base_dist = (
            base_dist
            if isinstance(base_dist, ConditionalDistribution)
            else ConstantConditionalDistribution(base_dist)
        )
        self.transforms = [
            (
                t
                if isinstance(t, ConditionalTransform)
                else ConstantConditionalTransform(t)
            )
            for t in transforms
        ]

    def condition(self, context):
        base_dist = self.base_dist.condition(context)
        transforms = [t.condition(context) for t in self.transforms]
        return TransformedDistribution(base_dist, transforms)

    def clear_cache(self):
        pass
