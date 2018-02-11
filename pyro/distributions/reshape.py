from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.util import sum_rightmost


class Reshape(Distribution, torch.distributions.Distribution):
    def __init__(self, base_dist, sample_shape=torch.Size(), extra_event_dim=0):
        sample_shape = torch.Size(sample_shape)
        self.base_dist = base_dist
        self.sample_shape = sample_shape
        self.extra_event_dim = extra_event_dim
        shape = sample_shape + base_dist.batch_shape + base_dist.event_shape
        batch_dim = len(shape) - extra_event_dim - len(base_dist.event_shape)
        batch_shape, event_shape = shape[:batch_dim], shape[batch_dim:]
        torch.distributions.Distribution.__init__(self, batch_shape, event_shape)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(self.sample_shape + sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(self.sample_shape + sample_shape)

    def log_pob(self, value):
        return sum_rightmost(self.base_dist.log_prob(value), self.extra_event_dim)

    def score_parts(self, value):
        log_pdf, score_function, entropy_term = self.base_dist.score_parts(value)
        log_pdf = sum_rightmost(log_pdf, self.extra_event_dim)
        score_function = sum_rightmost(score_function, self.extra_event_dim)
        entropy_term = sum_rightmost(entropy_term, self.extra_event_dim)
        return ScoreParts(log_pdf, score_function, entropy_term)

    def enumerate_support(self):
        samples = self.base_dist.enumerate_support()
        if not self.sample_shape:
            return samples
        enum_shape, base_shape = samples.shape[:1], samples.shape[1:]
        samples = samples.contiguous()
        samples = samples.view(enum_shape + (1,) * len(self.sample_shape) + base_shape)
        samples = samples.expand(enum_shape + self.sample_shape + base_shape)
        return samples

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance
