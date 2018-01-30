from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Binomial(TorchDistribution):
    """
    Binomial distribution.

    Distribution over counts for `n` independent `Bernoulli(ps)` trials.

    :param int n: Number of trials. Should be positive.
    :param torch.autograd.Variable ps: Probabilities. Should lie in the
        interval `[0,1]`.
    """

    def __init__(self, n, ps, *args, **kwargs):
        torch_dist = torch.distributions.Binomial(n, ps)
        super(Binomial, self).__init__(torch_dist, *args, **kwargs)
