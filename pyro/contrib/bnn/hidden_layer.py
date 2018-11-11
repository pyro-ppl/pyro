from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import Categorical
from torch.distributions.utils import lazy_property

from pyro.contrib.bnn.utils import adjoin_ones_vector
from pyro.distributions.torch_distribution import TorchDistribution


class HiddenLayer(TorchDistribution):
    r"""
    :param torch.Tensor X: B x D dimensional mini-batch of input
    :param torch.Tensor A_scale: D x H     dimensional (diagonal covariance) or
                                 D x D x H dimensional (cholesky factorization) or
                                 D x K x H dimensional (mixture of diagonal normals)
    :param torch.Tensor A_logits: H x K dimensional (for mixture of diagonal normals)
    """
    has_rsample = True

    def __init__(self, X=None, A_mean=None, A_scale=None, A_logits=None, non_linearity="relu",
                 KL_factor=1.0, A_prior_scale=1.0, include_hidden_bias=True,
                 weight_space_sampling=False, relu_shift=None, relu_scale=None, N_KL_samples=5000):
        self.X = X
        self.dim_X = X.size(-1)
        self.dim_H = A_mean.size(-1)
        # assert A_mean.shape == A_scale.shape
        #assert A_mean.size(0) == self.dim_X
        self.A_mean = A_mean
        self.A_scale = A_scale
        self.A_logits = A_logits
        self.non_linearity = non_linearity
        if A_scale.dim()==2:
            self.A_covariance = "diagonal"
        elif A_scale.dim() == 3 and self.A_logits is not None:
            self.A_covariance = "mixture"
            self.K = A_scale.size(-2)
            self.A_categorical = Categorical(logits=A_logits)
        elif A_scale.dim() == 3 and self.A_logits is None:
            self.A_covariance = "cholesky"
        self.KL_factor = KL_factor
        self.A_prior_scale = A_prior_scale
        self.weight_space_sampling = weight_space_sampling
        self.include_hidden_bias = include_hidden_bias
        self.leaky_epsilon = 0.1
        self.hard_sigmoid_alpha = 1.0
        self.relu_scale = relu_scale
        self.relu_shift = relu_shift
        if non_linearity == "flex_relu":
            assert relu_scale is not None and relu_shift is not None and \
                   relu_scale.size(-1) == self.dim_H and relu_shift.size(-1) == self.dim_H
        if non_linearity == "flex_relu_mean":
            assert relu_shift is not None and relu_shift.size(-1) == self.dim_H
        self.N_KL_samples = N_KL_samples
        assert non_linearity in ["linear", "relu", "leaky_relu", "hard_sigmoid", "flex_relu", "flex_relu_mean"], \
            "non-linearity must be one of: linear, relu, leaky_relu, hard_sigmoid"

    def log_prob(self, value):
        return -self.KL_factor * self.KL

    @lazy_property
    def KL(self):
        KL_approx = "var"
        # D x K x H dimensional (mixture of diagonal normals)
        if self.A_covariance=="mixture":
            probs = self.A_categorical.probs.t()  # K H
            if KL_approx=="sampling":
                A = HiddenLayer.rsample(self, sample_shape=(self.N_KL_samples,), return_A=True)  # NS D H
                A_shift = (A - self.A_mean).unsqueeze(-2)  # NS D 1 H
                exponent = -0.5 * torch.pow(A_shift / self.A_scale, 2.0).sum(1)  # NS K H
                max_exponent = torch.max(exponent, dim=1, keepdim=True)[0]  # NS 1 H
                A_scale_prod = self.A_scale.log().sum(0).exp()
                KL_A = torch.exp(exponent - max_exponent) / A_scale_prod  # NS K H
                KL_A = (KL_A * probs).sum(1).log()  # NS H
                KL_A += max_exponent.squeeze(1)  # NS H
                KL_A = 2.0 * KL_A.sum() / self.N_KL_samples
                #KL_A += 0.5 * torch.pow(A / self.A_prior_scale, 2.0).sum() / self.N_KL_samples
                #KL_A += math.log(self.A_prior_scale) * float(self.A_mean.shape[0] * self.A_mean.shape[1])
                KL_A += torch.pow(self.A_mean / self.A_prior_scale, 2.0).sum()
                A_scale_sqr = torch.pow(self.A_scale, 2.0)
                KL_A += (A_scale_sqr * probs).sum() / torch.pow(self.A_prior_scale, 2.0)
                KL_A += 2.0 * float(self.A_mean.shape[0] * self.A_mean.shape[1]) * math.log(self.A_prior_scale)
            elif KL_approx=="bound":
                KL_A = torch.pow(self.A_mean / self.A_prior_scale, 2.0).sum()
                A_scale_sqr = torch.pow(self.A_scale, 2.0)
                KL_A += (A_scale_sqr * probs).sum() / torch.pow(self.A_prior_scale, 2.0)
                KL_A += 2.0 * float(self.A_mean.shape[0] * self.A_mean.shape[1]) * math.log(self.A_prior_scale)
                covcov = (A_scale_sqr.unsqueeze(-2) + A_scale_sqr.unsqueeze(-3))  # D K K H
                sqrt_inv_det = -0.5 * covcov.log().sum(0)  # K K H
                max_exponent = torch.max(sqrt_inv_det, dim=-2, keepdim=True)[0]
                sqrt_inv_det = (sqrt_inv_det - max_exponent).exp()  # K K H
                log_term = (sqrt_inv_det * probs).sum(-2).log() + max_exponent.squeeze(-2)  # K H
                KL_A += 2.0 * (probs * log_term).sum()
            elif KL_approx=="var":
                KL_A = torch.pow(self.A_mean / self.A_prior_scale, 2.0).sum()
                A_scale_sqr = torch.pow(self.A_scale, 2.0)
                KL_A += (A_scale_sqr * probs).sum() / torch.pow(self.A_prior_scale, 2.0)
                KL_A += 2.0 * float(self.A_mean.shape[0] * self.A_mean.shape[1]) * math.log(self.A_prior_scale)

                Tr_ab = (A_scale_sqr.unsqueeze(-2) / A_scale_sqr.unsqueeze(-3)).sum(0)  # K K H
                log_scale_sum = self.A_scale.log().sum(0)  # K H
                log_det_ab = log_scale_sum.unsqueeze(-3) - log_scale_sum.unsqueeze(-2)  # K K H
                KL_ab = -0.5 * Tr_ab - log_det_ab  # K K H
                max_KL = torch.max(KL_ab, dim=-2, keepdim=True)[0] # K 1 H
                exp_KL_ab = torch.exp(KL_ab - max_KL) * probs  # K K H
                term_1 = exp_KL_ab.sum(-2).log() + max_KL.squeeze(-2)  # K H
                KL_A += 2.0 * (probs * term_1).sum()

                KL_A -= (log_scale_sum * probs).sum()
        else:
            KL_A = torch.pow(self.A_mean / self.A_prior_scale, 2.0).sum()
            KL_A -= self.dim_X * self.dim_H
            KL_A += torch.pow(self.A_scale / self.A_prior_scale, 2.0).sum()
            if self.A_covariance=='diagonal':
                KL_A -= 2.0 * torch.log(self.A_scale / self.A_prior_scale).sum()
            elif self.A_covariance=='cholesky':
                KL_A -= 2.0 * (torch.diagonal(self.A_scale, dim1=-3, dim2=-2).log() - self.A_prior_scale.log()).sum()
        return 0.5 * KL_A

    def rsample(self, sample_shape=torch.Size(), return_activation=False, return_A=False):
        if self.A_covariance=="mixture":
            which = self.A_categorical.sample(sample_shape).unsqueeze(-2).unsqueeze(-2)
            which = which.expand(sample_shape + (self.dim_X, 1, self.dim_H))
            A_scale_expand = self.A_scale.unsqueeze(0).expand(sample_shape + self.A_scale.shape)
            A_scale_which = torch.gather(A_scale_expand, -2, which).squeeze(-2)

            if self.weight_space_sampling or return_A:
                A = self.A_mean + torch.randn(A_scale_which.shape).type_as(self.A_mean) * A_scale_which
                if return_A:
                    return A
                activation = torch.matmul(self.X, A)
            else:
                _mean = torch.matmul(self.X, self.A_mean)
                X_sqr = torch.pow(self.X, 2.0).unsqueeze(-1)
                A_scale_sqr = torch.pow(A_scale_which, 2.0).unsqueeze(-3)
                _std = (X_sqr * A_scale_sqr).sum(-2).sqrt()
                activation = _mean + torch.randn(_std.shape).type_as(_std) * _std

        if self.weight_space_sampling:
            if self.A_covariance=='diagonal':
                A = self.A_mean + torch.randn(sample_shape + self.A_scale.shape).type_as(self.A_mean) * self.A_scale
            elif self.A_covariance=='cholesky':
                eps = torch.randn(sample_shape + (1, self.dim_H, self.dim_X)).type_as(self.A_mean)
                A = torch.matmul(eps, self.A_scale)
                A = torch.diagonal(A, dim1=-2, dim2=-1) + self.A_mean
            activation = torch.matmul(self.X, A)
        else:
            _mean = torch.matmul(self.X, self.A_mean)
            if self.A_covariance=='diagonal':
                X_sqr = torch.pow(self.X, 2.0).unsqueeze(-1)
                A_scale_sqr = torch.pow(self.A_scale, 2.0)
                _std = (X_sqr * A_scale_sqr).sum(-2).sqrt()
            elif self.A_covariance=='cholesky':
                A_scale_trans = torch.transpose(self.A_scale, 0, 1)  # D x D x H
                X_L_jbh = torch.matmul(self.X, A_scale_trans)  # D x B x H
                X_L_sqr_bh = torch.pow(X_L_jbh, 2.0).sum(0)  # B x H
                _std = X_L_sqr_bh.sqrt()
            activation = _mean + torch.randn(sample_shape + _std.shape).type_as(_std) * _std
        if self.non_linearity == "relu":
            activation = torch.nn.functional.relu(activation)
        elif self.non_linearity =="flex_relu":
            activation = self.relu_scale * torch.nn.functional.relu(activation - self.relu_shift)
        elif self.non_linearity =="flex_relu_mean":
            activation = torch.nn.functional.relu(activation - self.relu_shift)
        elif self.non_linearity == "leaky_relu":
            activation = torch.nn.functional.leaky_relu(activation, negative_slope=self.leaky_epsilon)
        elif self.non_linearity == "hard_sigmoid":
            plus_alpha = torch.abs(activation + self.hard_sigmoid_alpha)
            minus_alpha = torch.abs(activation - self.hard_sigmoid_alpha)
            activation = 0.5 * (plus_alpha - minus_alpha)
        if self.include_hidden_bias:
            activation = adjoin_ones_vector(activation)
        return activation
