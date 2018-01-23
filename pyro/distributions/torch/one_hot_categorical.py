from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class OneHotCategorical(TorchDistribution):
    """
    OneHotCategorical (discrete) distribution.

    Discrete distribution over one-hot vectors.

    :param ps: Probabilities. These should be non-negative and normalized
        along the rightmost axis.
    :type ps: torch.autograd.Variable
    :param logits: Log probability values. When exponentiated, these should
        sum to 1 along the last axis. Either `ps` or `logits` should be
        specified but not both.
    :type logits: torch.autograd.Variable
    :param batch_size: Optional number of elements in the batch used to
        generate a sample. The batch dimension will be the leftmost dimension
        in the generated sample.
    :type batch_size: int
    """
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.OneHotCategorical(probs=ps, logits=logits)
        super(OneHotCategorical, self).__init__(torch_dist, *args, **kwargs)

    def sample(self, sample_shape=torch.Size()):
        """
        Returns a sample which has the same shape as `ps`, except that the last dimension
        will have the same size as the number of events.

        :return: sample from the OneHotCategorical distribution
        :rtype: torch.Tensor
        """
        return super(OneHotCategorical, self).sample(sample_shape)

    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for one or a batch of samples and
        parameters.  The last dimension for `ps` encodes the event
        probabilities, and the remaining dimensions are considered batch
        dimensions.

        `ps` and first broadcasted to the size of the data `x`. The data tensor
        is used to to create a mask over `ps` where a 1 in the mask indicates
        that the corresponding probability in `ps` was selected. The method
        returns the logarithm of these probabilities.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        return super(OneHotCategorical, self).batch_log_pdf(x)

    def enumerate_support(self):
        """
        Returns the categorical distribution's support, as a tensor along the
        first dimension.

        Note that this returns support values of all the batched RVs in
        lock-step, rather than the full cartesian product. To iterate over the
        cartesian product, you must construct univariate Categoricals and use
        itertools.product() over all univariate variables (but this is very
        expensive).

        :param torch.autograd.Variable ps: Tensor where the last dimension
            denotes the event probabilities, *p_k*, which must sum to 1. The
            remaining dimensions are considered batch dimensions.
        :return: Torch variable enumerating the support of the categorical
            distribution.  Each item in the return value, when enumerated along
            the first dimensions, yields a value from the distribution's
            support which has the same dimension as would be returned by
            sample. The last dimension is used for the one-hot encoding.
        :rtype: torch.autograd.Variable.
        """
        return super(OneHotCategorical, self).enumerate_support()
