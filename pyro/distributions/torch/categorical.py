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
    :param batch_size: Optional number of elements in the batch used to
        generate a sample. The batch dimension will be the leftmost dimension
        in the generated sample.
    :type batch_size: int
    """
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.Categorical(probs=ps, logits=logits)
        super(Categorical, self).__init__(torch_dist, *args, **kwargs)

    def sample(self, sample_shape=torch.Size()):
        """
        Returns a sample which has the same shape as `ps` (or `vs`). The type
        of the sample is `numpy.ndarray` if `vs` is a list or a numpy array,
        else a tensor is returned.

        :return: sample from the Categorical distribution
        :rtype: numpy.ndarray or torch.LongTensor
        """
        return super(Categorical, self).sample(sample_shape)

    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for one or a batch of samples and parameters.
        The last dimension for `ps` encodes the event probabilities, and the remaining
        dimensions are considered batch dimensions.

        `ps` and `vs` are first broadcasted to the size of the data `x`. The
        data tensor is used to to create a mask over `vs` where a 1 in the mask
        indicates that the corresponding value in `vs` was selected. Since, `ps`
        and `vs` have the same size, this mask when applied over `ps` gives
        the probabilities of the selected events. The method returns the logarithm
        of these probabilities.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        return super(Categorical, self).batch_log_pdf(x)

    def enumerate_support(self):
        """
        Returns the categorical distribution's support, as a tensor along the first dimension.

        Note that this returns support values of all the batched RVs in lock-step, rather
        than the full cartesian product. To iterate over the cartesian product, you must
        construct univariate Categoricals and use itertools.product() over all univariate
        variables (but this is very expensive).

        :param ps: Tensor where the last dimension denotes the event probabilities, *p_k*,
            which must sum to 1. The remaining dimensions are considered batch dimensions.
        :type ps: torch.autograd.Variable
        :param vs: Optional parameter, enumerating the items in the support. This could either
            have a numeric or string type. This should have the same dimension as ``ps``.
        :type vs: list or numpy.ndarray or torch.autograd.Variable
        :return: Torch variable or numpy array enumerating the support of the categorical distribution.
            Each item in the return value, when enumerated along the first dimensions, yields a
            value from the distribution's support which has the same dimension as would be returned by
            sample.
        :rtype: torch.autograd.Variable or numpy.ndarray.
        """
        return super(Categorical, self).enumerate_support()
