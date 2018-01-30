from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class OneHotCategorical(TorchDistribution):
    """
    OneHotCategorical (discrete) distribution, over one-hot vectors.

    :param ps: Probabilities. These should be non-negative and normalized
        along the rightmost axis.
    :type ps: torch.autograd.Variable
    :param logits: Log probability values. When exponentiated, these should
        sum to 1 along the last axis. Either `ps` or `logits` should be
        specified but not both.
    :type logits: torch.autograd.Variable
    """
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.OneHotCategorical(probs=ps, logits=logits)
        super(OneHotCategorical, self).__init__(torch_dist, *args, **kwargs)
