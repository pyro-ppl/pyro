from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch import Gamma, Independent, Normal


class SigmaPointMixin(object):
    def __init__(self, *args, **kwargs):
        self._importance_weights = {}
        super(SigmaPointMixin, self).__init__(*args, **kwargs)

    def log_prob(self, value):
        result = super(SigmaPointMixin, self).log_prob(value)
        if value in self._importance_weights:
            # this simulates a poutine.scale
            result = result * self._importance_weights[value]
        return result


class SigmaPointNormal(SigmaPointMixin, Normal):
    has_enumerate_support = True

    def enumerate_support(self, expand=True):
        if expand:
            return torch.stack([self.loc - self.scale,
                                self.loc + self.scale])
        # TODO heuristically reduce size of set of values
        value = torch.stack([self.loc - self.scale,
                             self.loc + self.scale])
        value = value.reshape((-1,) + (1,) * len(self.batch_shape))
        self._importance_weights[value] = torch.eye(value.size(0)).reshape((-1,) + self.batch_shape)


class SigmaPointDirichlet(SigmaPointMixin, Independent):
    has_enumerate_support = True

    def __init__(self, concentration):
        assert concentration.dim() >= 1
        base_dist = Gamma(concentration, 1.0)
        super(SigmaPointDirichlet, self).__init__(base_dist, 1)

    def enumerate_support(self, expand=True):
        if expand:
            return self.base_dist.alpha.unsqueeze(0)
        raise NotImplementedError
