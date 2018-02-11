from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(torch.distributions.Normal)
@copy_docs_from(torch.distributions.Distribution)
@copy_docs_from(Distribution)
class Normal(Distribution, torch.distribution.Normal):
    def __init__(self, mu, sigma):
        torch.distributions.Normal.__init__(mu, sigma)
