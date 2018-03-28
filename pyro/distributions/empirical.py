from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from pyro.distributions.distribution import Distribution


class Empirical(Distribution):
    r"""
    Empirical distribution associated with the sampled data. This is
    used for assessing the statistical properties of the sample.
    In particular, scoring of additional data points from the sample
    via ``log_prob`` is not currently supported.
    """

    def __init__(self):
        self.samples = []
        self.tensorized_samples = None
        super(Distribution, self).__init__()

    @property
    def finalized_tensor(self):
        """
        Collect samples along the left-most dimension and return the
        tensor.

        :return: torch.Tensor
        """
        if not self.samples or self.tensorized_samples is None:
            return self.tensorized_samples
        new_samples = torch.stack(self.samples, dim=0)
        self.tensorized_samples = torch.cat([self.tensorized_samples, new_samples], dim=0)
        self.samples = []
        return self.tensorized_samples

    @property
    def sample_size(self):
        return self.finalized_tensor.size(0)

    def add(self, value):
        """
        Adds the data point to the sample. The values in successive
        calls to ``add`` must have the same tensor shape and size.

        :param torch.Tensor value: tensor to add to the sample.
        """
        if self.tensorized_samples is None:
            self.tensorized_samples = value.new(0)
        self.samples.append(value)

    def sample(self, sample_shape=torch.Size()):
        if not sample_shape:
            rand_idx = np.random.randint(0, high=self.finalized_tensor.size(0), dtype=np.int64)
        else:
            rand_idx = np.random.randint(0, high=self.finalized_tensor.size(0), dtype=np.int64,
                                         size=np.prod(sample_shape).astype(np.int64))
        return self.finalized_tensor[rand_idx].view(sample_shape + self.batch_shape)

    def log_prob(self, value):
        raise NotImplemented("Empirical distribution does not implement `log_prob`")

    @property
    def batch_shape(self):
        if self.finalized_tensor is None:
            return None
        return self.finalized_tensor.size()[1:]

    @property
    def mean(self):
        samples = self.finalized_tensor.type(torch.float32) \
            if self.finalized_tensor.dtype in (torch.int, torch.long) else self.finalized_tensor
        return samples.mean(dim=0)

    @property
    def variance(self):
        samples = self.finalized_tensor.type(torch.float32) \
            if self.finalized_tensor.dtype in (torch.int, torch.long) else self.finalized_tensor
        return samples.var(dim=0)
