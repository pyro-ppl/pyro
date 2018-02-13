from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution


class TransformedDistribution(Distribution, torch.distributions.TransformedDistribution):
    def __init__(self, base_dist, transforms):
        torch.distributions.TransformedDistribution.__init__(self, base_dist, transforms)
