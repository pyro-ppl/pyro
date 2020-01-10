# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions.utils import lazy_property
import torch.nn.functional as F

from pyro.contrib.bnn.utils import adjoin_ones_vector
from pyro.distributions.torch_distribution import TorchDistribution


class HiddenLayer(TorchDistribution):
    r"""
    This distribution is a basic building block in a Bayesian neural network.
    It represents a single hidden layer, i.e. an affine transformation applied
    to a set of inputs `X` followed by a non-linearity. The uncertainty in the
    weights is encoded in a Normal variational distribution specified by the
    parameters `A_scale` and `A_mean`. The so-called 'local reparameterization
    trick' is used to reduce variance (see reference below). In effect, this
    means the weights are never sampled directly; instead one samples in
    pre-activation space (i.e. before the non-linearity is applied). Since the
    weights are never directly sampled, when this distribution is used within
    the context of variational inference, care must be taken to correctly scale
    the KL divergence term that corresponds to the weight matrix. This term is
    folded into the `log_prob` method of this distributions.

    In effect, this distribution encodes the following generative process:

    A ~ Normal(A_mean, A_scale)
    output ~ non_linearity(AX)

    :param torch.Tensor X: B x D dimensional mini-batch of inputs
    :param torch.Tensor A_mean:  D x H dimensional specifiying weight mean
    :param torch.Tensor A_scale: D x H dimensional (diagonal covariance matrix)
                                 specifying weight uncertainty
    :param callable non_linearity: a callable that specifies the
                                   non-linearity used. defaults to ReLU.
    :param float KL_factor: scaling factor for the KL divergence. prototypically
                            this is equal to the size of the mini-batch divided
                            by the size of the whole dataset. defaults to `1.0`.
    :param A_prior: the prior over the weights is assumed to be normal with
                    mean zero and scale factor `A_prior`. default value is 1.0.
    :type A_prior: float or torch.Tensor
    :param bool include_hidden_bias: controls whether the activations should be
                                     augmented with a 1, which can be used to
                                     incorporate bias terms. defaults to `True`.
    :param bool weight_space_sampling: controls whether the local reparameterization
                                       trick is used. this is only intended to be
                                       used for internal testing.
                                       defaults to `False`.

    Reference:

    Kingma, Diederik P., Tim Salimans, and Max Welling.
    "Variational dropout and the local reparameterization trick."
    Advances in Neural Information Processing Systems. 2015.
    """
    has_rsample = True

    def __init__(self, X=None, A_mean=None, A_scale=None, non_linearity=F.relu,
                 KL_factor=1.0, A_prior_scale=1.0, include_hidden_bias=True,
                 weight_space_sampling=False):
        self.X = X
        self.dim_X = X.size(-1)
        self.dim_H = A_mean.size(-1)
        assert A_mean.size(0) == self.dim_X, \
            "The dimensions of X and A_mean and A_scale must match accordingly; see documentation"
        self.A_mean = A_mean
        self.A_scale = A_scale
        self.non_linearity = non_linearity
        assert callable(non_linearity), "non_linearity must be callable"
        if A_scale.dim() != 2:
            raise NotImplementedError("A_scale must be 2-dimensional")

        self.KL_factor = KL_factor
        self.A_prior_scale = A_prior_scale
        self.weight_space_sampling = weight_space_sampling
        self.include_hidden_bias = include_hidden_bias

    def log_prob(self, value):
        return -self.KL_factor * self.KL

    @lazy_property
    def KL(self):
        KL_A = torch.pow(self.A_mean / self.A_prior_scale, 2.0).sum()
        KL_A -= self.dim_X * self.dim_H
        KL_A += torch.pow(self.A_scale / self.A_prior_scale, 2.0).sum()
        KL_A -= 2.0 * torch.log(self.A_scale / self.A_prior_scale).sum()
        return 0.5 * KL_A

    def rsample(self, sample_shape=torch.Size()):
        # note: weight space sampling is only meant for testing
        if self.weight_space_sampling:
            A = self.A_mean + torch.randn(sample_shape + self.A_scale.shape).type_as(self.A_mean) * self.A_scale
            activation = torch.matmul(self.X, A)
        else:
            _mean = torch.matmul(self.X, self.A_mean)
            X_sqr = torch.pow(self.X, 2.0).unsqueeze(-1)
            A_scale_sqr = torch.pow(self.A_scale, 2.0)
            _std = (X_sqr * A_scale_sqr).sum(-2).sqrt()
            activation = _mean + torch.randn(sample_shape + _std.shape).type_as(_std) * _std

        # apply non-linearity
        activation = self.non_linearity(activation)

        # add 1 element to activations
        if self.include_hidden_bias:
            activation = adjoin_ones_vector(activation)

        return activation
