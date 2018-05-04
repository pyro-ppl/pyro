from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch import RelaxedOneHotCategorical
from pyro.distributions.util import copy_docs_from


@copy_docs_from(RelaxedOneHotCategorical)
class RelaxedCategoricalStraightThrough(RelaxedOneHotCategorical):
    """
    Implementation of ``Dirichlet`` via ``Gamma``.

    This naive implementation has stochastic reparameterized gradients, which
    have higher variance than PyTorch's ``Dirichlet`` implementation.
    """
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super(RelaxedCategoricalStraightThrough, self).__init__(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args)
        self._unquantize = {}

    def rsample(self, sample_shape=torch.Size()):
        soft_sample = super(RelaxedCategoricalStraightThrough, self).rsample(sample_shape)
        hard_sample = QuantizeCategorical.apply(soft_sample)
        self._unquantize[id(hard_sample)] = soft_sample
        return hard_sample

    def log_prob(self, value):
        value = self._unquantize.get(id(value), value)
        return super(RelaxedCategoricalStraightThrough, self).log_prob(value)


class QuantizeCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        argmax = input.max(-1)[1]
        result = input.new_zeros(input.shape)
        # result[argmax] = 1
        if argmax.dim() < result.dim():
            argmax = argmax.unsqueeze(-1)
        return result.scatter_(-1, argmax, 1)
        #import pdb as pdb; pdb.set_trace()
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


# class RelaxedBernoulliStraightThrough(RelaxedBernoulli):
#     def rsample(self, *args, **kwargs):
#         sample = super(RelaxedBernoulliStraightThrough, self).rsample(*args, **kwargs)
#         return Round.apply(sample)

# class QuantizeBernoulli(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return torch.round(input)

#     @staticmethod
#     def backward(ctx, grad):
#         return grad.clone()