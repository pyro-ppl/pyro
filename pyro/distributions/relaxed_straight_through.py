# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions.utils import clamp_probs

from pyro.distributions.torch import RelaxedBernoulli, RelaxedOneHotCategorical
from pyro.distributions.util import copy_docs_from


@copy_docs_from(RelaxedOneHotCategorical)
class RelaxedOneHotCategoricalStraightThrough(RelaxedOneHotCategorical):
    """
    An implementation of
    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
    with a straight-through gradient estimator.

    This distribution has the following properties:

    - The samples returned by the :meth:`rsample` method are discrete/quantized.
    - The :meth:`log_prob` method returns the log probability of the
      relaxed/unquantized sample using the GumbelSoftmax distribution.
    - In the backward pass the gradient of the sample with respect to the
      parameters of the distribution uses the relaxed/unquantized sample.

    References:

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables,
        Chris J. Maddison, Andriy Mnih, Yee Whye Teh
    [2] Categorical Reparameterization with Gumbel-Softmax,
        Eric Jang, Shixiang Gu, Ben Poole
    """
    def rsample(self, sample_shape=torch.Size()):
        soft_sample = super().rsample(sample_shape)
        soft_sample = clamp_probs(soft_sample)
        hard_sample = QuantizeCategorical.apply(soft_sample)
        return hard_sample

    def log_prob(self, value):
        value = getattr(value, '_unquantize', value)
        return super().log_prob(value)


class QuantizeCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_value):
        argmax = soft_value.max(-1)[1]
        hard_value = torch.zeros_like(soft_value)
        hard_value._unquantize = soft_value
        if argmax.dim() < hard_value.dim():
            argmax = argmax.unsqueeze(-1)
        return hard_value.scatter_(-1, argmax, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad


@copy_docs_from(RelaxedBernoulli)
class RelaxedBernoulliStraightThrough(RelaxedBernoulli):
    """
    An implementation of
    :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli`
    with a straight-through gradient estimator.

    This distribution has the following properties:

    - The samples returned by the :meth:`rsample` method are discrete/quantized.
    - The :meth:`log_prob` method returns the log probability of the
      relaxed/unquantized sample using the GumbelSoftmax distribution.
    - In the backward pass the gradient of the sample with respect to the
      parameters of the distribution uses the relaxed/unquantized sample.

    References:

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables,
        Chris J. Maddison, Andriy Mnih, Yee Whye Teh
    [2] Categorical Reparameterization with Gumbel-Softmax,
        Eric Jang, Shixiang Gu, Ben Poole
    """
    def rsample(self, sample_shape=torch.Size()):
        soft_sample = super().rsample(sample_shape)
        soft_sample = clamp_probs(soft_sample)
        hard_sample = QuantizeBernoulli.apply(soft_sample)
        return hard_sample

    def log_prob(self, value):
        value = getattr(value, '_unquantize', value)
        return super().log_prob(value)


class QuantizeBernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_value):
        hard_value = soft_value.round()
        hard_value._unquantize = soft_value
        return hard_value

    @staticmethod
    def backward(ctx, grad):
        return grad
