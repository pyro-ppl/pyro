from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch import RelaxedOneHotCategorical
from pyro.distributions.util import copy_docs_from


@copy_docs_from(RelaxedOneHotCategorical)
class RelaxedOneHotCategoricalStraightThrough(RelaxedOneHotCategorical):
    """
    An implementation of RelaxedOneHotCategorical with a straight-through gradient estimator.

    This distribution has the following properties:
    -- the samples returned by the `rsample` method are discrete/quantized
    -- the `log_prob` method returns the log probability of the relaxed/unquantized sample
       using the GumbelSoftmax distribution
    -- in the backward pass the gradient of the sample with respect to the parameters of the
       distribution uses the relaxed/unquantized sample

    References:

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables,
        Chris J. Maddison, Andriy Mnih, Yee Whye Teh
    [2] Categorical Reparameterization with Gumbel-Softmax,
        Eric Jang, Shixiang Gu, Ben Poole
    """
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super(RelaxedOneHotCategoricalStraightThrough, self).__init__(temperature=temperature, probs=probs,
                                                                      logits=logits, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        soft_sample = super(RelaxedOneHotCategoricalStraightThrough, self).rsample(sample_shape)
        hard_sample = QuantizeCategorical.apply(soft_sample)
        return hard_sample

    def log_prob(self, value):
        value = getattr(value, '_unquantize', value)
        return super(RelaxedOneHotCategoricalStraightThrough, self).log_prob(value)


class QuantizeCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_value):
        argmax = soft_value.max(-1)[1]
        hard_value = soft_value.new_zeros(soft_value.shape)
        hard_value._unquantize = soft_value
        if argmax.dim() < hard_value.dim():
            argmax = argmax.unsqueeze(-1)
        return hard_value.scatter_(-1, argmax, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad
