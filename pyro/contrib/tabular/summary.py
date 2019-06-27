from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import torch
from six import add_metaclass


@add_metaclass(ABCMeta)
class Summary(object):
    """
    Abstract base class for collated sufficient statistics of data.

    When fitting components of a mixture model, and when data-to-component
    assignments are known, then a summary should be as good as the real dataset.
    """

    @abstractmethod
    def scatter_update(self, component, data):
        """
        Add data to this summary, collating according to known components.

        :param torch.Tensor component: A column of component indices.
        :param torch.Tensor data: A column of data.
        """
        pass

    @abstractmethod
    def __imul__(self, scale):
        """
        Scale this summary by a factor, used for exponential forgetting.

        :param float scale: A scale factor, typically in ``(0,1)``.
        """
        pass

    @abstractmethod
    def as_scaled_data(self):
        """
        Create a pseudo dataset that is indistinguishable from the original
        summarized dataset from the point of view of the feature model.

        :return: A pair ``scale, data`` where ``scale`` is a scaling tensor
            and ``data`` is a batch of data with batch shape
            ``(num_components, data_size)``. Implementation may choose
            arbitrary ``data_size``.
        :rtype: tuple
        """
        pass


class BernoulliSummary(Summary):
    def __init__(self, num_components, prototype):
        self.counts = prototype.new_zeros(num_components, 2)

    def scatter_update(self, component, data):
        self.counts[:, 0].scatter_add_(0, component, 1 - data)
        self.counts[:, 1].scatter_add_(0, component, data)

    def __imul__(self, scale):
        self.counts *= scale

    def as_scaled_data(self):
        scale = self.counts
        data = scale.new_tensor([0., 1.])
        return scale, data


class NormalSummary(Summary):
    """
    Summary of univariate Normal data.
    """
    def __init__(self, num_components, prototype):
        self.count = prototype.new_zeros(num_components)
        self.mean = prototype.new_zeros(num_components)
        self.count_times_variance = prototype.new_zeros(num_components)
        super(NormalSummary, self).__init__()

    def scatter_update(self, component, data):
        assert component.dim() == 1
        assert data.shape == component.shape
        count = torch.full_like(self.count, 1e-20).scatter_add_(0, component, torch.ones_like(data))
        mean = torch.zeros_like(self.mean).scatter_add_(0, component, data) / count
        delta = count / (count + self.count) * (mean - self.mean)
        self.mean += delta
        self.count_times_variance += self.count * delta.pow(2)
        self.count_times_variance.scatter_add_(0, component, (data - self.mean[component]).pow(2))
        self.count += count

    def __imul__(self, scale):
        self.count *= scale
        self.count_times_variance *= scale

    def as_scaled_data(self):
        scale = (self.count_times_variance / self.count).sqrt()
        data = torch.stack([self.mean - scale, self.mean + scale], dim=-1)
        scale = self.count.unsqueeze(-1) * 0.5
        return scale, data
