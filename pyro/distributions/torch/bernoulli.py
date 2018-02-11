from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Bernoulli(TorchDistribution):
    """
    Bernoulli distribution.

    Distribution over a vector of independent Bernoulli variables. Each element
    of the vector takes on a value in `{0, 1}`.

    This is often used in conjunction with `torch.nn.Sigmoid` to ensure the
    `ps` parameters are in the interval `[0, 1]`.

    :param torch.autograd.Variable ps: Probabilities. Should lie in the
        interval `[0,1]`.
    :param logits: Log odds, i.e. :math:`\\log(\\frac{p}{1 - p})`. Either `ps` or
        `logits` should be specified, but not both.
    """

    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.Bernoulli(probs=ps, logits=logits)
        super(Bernoulli, self).__init__(torch_dist, *args, **kwargs)
