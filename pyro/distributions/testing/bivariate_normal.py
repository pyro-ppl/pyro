from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from pyro.distributions.util import copy_docs_from
from pyro.distributions.distribution import Distribution


@copy_docs_from(Distribution)
class BivariateNormal(Distribution):
    def __init__(self, loc, scale_tril):
        self.loc = loc
        self.scale_tril = scale_tril

    def sample(self):
        return self.loc + torch.mv(self.scale_tril, self.loc.new(self.loc.shape).normal_())

    def batch_log_pdf(self, x):
        delta = x - self.loc
        z0 = delta[..., 0] / self.scale_tril[..., 0, 0]
        z1 = (delta[..., 1] - self.scale_tril[..., 1, 0] * z0) / self.scale_tril[..., 1, 1]
        z = torch.stack([z0, z1], dim=-1)
        mahalanobis_squared = (z ** 2).sum(-1)
        normalization_constant = self.scale_tril.diag().log().sum(-1) + np.log(2 * np.pi)
        return -(normalization_constant + 0.5 * mahalanobis_squared).unsqueeze(-1)
