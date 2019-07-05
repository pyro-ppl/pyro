from abc import abstractmethod

import torch


class ConditionalDistribution(object):
    @abstractmethod
    def condition(self, context):
        """:rtype: torch.distributions.Distribution"""
        raise NotImplementedError


class ConditionalTransform(object):
    @abstractmethod
    def condition(self, context):
        """:rtype: torch.distributions.Transform"""
        raise NotImplementedError


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
