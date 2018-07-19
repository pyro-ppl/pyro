from __future__ import absolute_import, division, print_function

import torch
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution


class MaskedMixture(TorchDistribution):
    """
    A masked deterministic mixture of two distributions.

    This is useful when the mask is sampled from another distribution,
    possibly correlated across the batch. Often the mask can be
    marginalized out via enumeration.

    Example::

        change_point = pyro.sample("change_point",
                                   dist.Categorical(torch.ones(len(data) + 1)),
                                   infer={'enumerate': 'parallel'})
        mask = torch.arange(len(data), dtype=torch.long) >= changepoint
        with pyro.iarange("data", len(data)):
            pyro.sample("obs", MaskedMixture(mask, dist1, dist2), obs=data)

    Note that :ivar:`support` is set to ``component0.support`` due to lack of
    programmatic ability to union constraints. To correctly set support of an
    instance, simply overwrite its :ivar:`support` attribute.

    :param torch.Tensor mask: A byte tensor toggling between ``component0``
        and ``component1``.
    :param pyro.distributions.TorchDistribution component0: a distribution
    :param pyro.distributions.TorchDistribution component1: a distribution
    """
    def __init__(self, mask, component0, component1):
        if component0.batch_shape != mask.shape:
            raise ValueError('component0 does not match mask shape: {} vs {}'.format(
                             component0.batch_shape, mask.shape))
        if component1.batch_shape != mask.shape:
            raise ValueError('component1 does not match mask shape: {} vs {}'
                             .format(component1.batch_shape, mask.shape))
        if component0.event_shape != component1.event_shape:
            raise ValueError('components event_shape disagree: {} vs {}'
                             .format(component0.event_shape, component1.event_shape))
        self.mask = mask
        self.component0 = component0
        self.component1 = component1
        self.support = component0.support  # best guess; may be incorrect
        super(MaskedMixture, self).__init__(component0.batch_shape, component0.event_shape)

    @property
    def has_rsample(self):
        return self.component0.has_rsample and self.component1.has_rsample

    def expand(self, batch_shape):
        try:
            return super(MaskedMixture, self).expand(batch_shape)
        except NotImplementedError:
            mask = self.mask.expand(batch_shape)
            components0 = self.components0.expand(batch_shape)
            components1 = self.components1.expand(batch_shape)
            result = type(self)(mask, components0, components1)
            result.support = self.support
            return result

    def sample(self, sample_shape=torch.Size()):
        mask = self.mask.expand(sample_shape + self.batch_shape) if sample_shape else self.mask
        result = self.component0.sample(sample_shape)
        result[mask] = self.component1.sample(sample_shape)[mask]
        return result

    def rsample(self, sample_shape=torch.Size()):
        mask = self.mask.expand(sample_shape + self.batch_shape) if sample_shape else self.mask
        result = self.component0.rsample(sample_shape)
        result[mask] = self.component1.rsample(sample_shape)[mask]
        return result

    def log_prob(self, value):
        result = self.component0.log_prob(value)
        result[self.mask] = self.component1.log_prob(value)[self.mask]
        return result

    @lazy_property
    def mean(self):
        result = self.component0.mean.clone()
        result[self.mask] = self.component1.mean[self.mask]
        return result

    @lazy_property
    def variance(self):
        result = self.component0.variance.clone()
        result[self.mask] = self.component1.variance[self.mask]
        return result
