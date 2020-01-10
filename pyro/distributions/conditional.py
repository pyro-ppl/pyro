# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
import torch.nn


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
        super(ConditionalTransformModule, self).__init__(*args, **kwargs)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()


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


class ConditionalTransformedDistribution(ConditionalDistribution):
    def __init__(self, base_dist, transforms):
        self.base_dist = base_dist if isinstance(
            base_dist, ConditionalDistribution) else ConstantConditionalDistribution(base_dist)
        self.transforms = [t if isinstance(t, ConditionalTransform)
                           else ConstantConditionalTransform(t) for t in transforms]

    def condition(self, context):
        base_dist = self.base_dist.condition(context)
        transforms = [t.condition(context) for t in self.transforms]
        return torch.distributions.TransformedDistribution(base_dist, transforms)
