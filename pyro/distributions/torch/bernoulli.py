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
    :param batch_size: The number of elements in the batch used to generate
        a sample. The batch dimension will be the leftmost dimension in the
        generated sample.
    :param log_pdf_mask: Tensor that is applied to the batch log pdf values
        as a multiplier. The most common use case is supplying a boolean
        tensor mask to mask out certain batch sites in the log pdf computation.
    """

    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.Bernoulli(probs=ps, logits=logits)
        super(Bernoulli, self).__init__(torch_dist, *args, **kwargs)
