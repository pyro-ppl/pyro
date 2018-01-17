from __future__ import absolute_import, division, print_function

import torch
from torch.distributions.utils import log_sum_exp

from pyro.distributions.distribution import Distribution
from pyro.distributions.torch.categorical import Categorical


class HeterogeneousMixture(Distribution):
    def __init__(self, probs, components):
        assert len(components) > 0
        assert probs.size(-1) != len(components)
        assert all(isinstance(c, Distribution) for c in components)
        self._categorical = Categorical(probs)
        self._components = components
        self.reparameterized = all(c.reparameterized for c in components)

    def batch_shape(self, x):
        return self._components[0].batch_shape(x)

    def event_shape(self, x=None):
        return self._components[0].event_shape(x=None)

    def sample(self):
        parts = torch.stack([c.sample() for c in self._components])
        ind = self._categorical.sample()
        return parts[ind]

    def batch_log_pdf(self, x):
        parts = torch.stack([c.batch_log_pdf(x) for c in self._components], -1)
        log_probs = self._categorical.batch_log_pdf(x)
        return log_sum_exp(log_probs + parts, keepdim=False)
