from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Categorical(TorchDistribution):
    """
    Categorical (discrete) distribution.

    Discrete distribution over elements of `vs` with :math:`P(vs[i]) \\propto ps[i]`.
    ``.sample()`` returns an element of ``vs`` category index.

    :param ps: Probabilities. These should be non-negative and normalized
        along the rightmost axis.
    :type ps: torch.autograd.Variable
    :param logits: Log probability values. When exponentiated, these should
        sum to 1 along the last axis. Either `ps` or `logits` should be
        specified but not both.
    :type logits: torch.autograd.Variable
    :param vs: Optional list of values in the support.
    :type vs: list or numpy.ndarray or torch.autograd.Variable
    :type batch_size: int
    """
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.Categorical(probs=ps, logits=logits)
        super(Categorical, self).__init__(torch_dist, *args, **kwargs)
