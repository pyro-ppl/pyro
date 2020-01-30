# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import Categorical, constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import sum_leftmost


class MixtureOfDiagNormalsSharedCovariance(TorchDistribution):
    """
    Mixture of Normal distributions with diagonal covariance matrices.

    That is, this distribution is a mixture with K components, where each
    component distribution is a D-dimensional Normal distribution with a
    D-dimensional mean parameter loc and a D-dimensional diagonal covariance
    matrix specified by a scale parameter `coord_scale`. The K different
    component means are gathered into the parameter `locs` and the scale
    parameter is shared between all K components. The mixture weights are
    controlled by a K-dimensional vector of softmax logits, `component_logits`.
    This distribution implements pathwise derivatives for samples from the
    distribution.

    See reference [1] for details on the implementations of the pathwise
    derivative. Please consider citing this reference if you use the pathwise
    derivative in your research. Note that this distribution does not support
    dimension D = 1.

    [1] Pathwise Derivatives for Multivariate Distributions, Martin Jankowiak &
    Theofanis Karaletsos. arXiv:1806.01856

    :param torch.Tensor locs: K x D mean matrix
    :param torch.Tensor coord_scale: shared D-dimensional scale vector
    :param torch.Tensor component_logits: K-dimensional vector of softmax logits
    """
    has_rsample = True
    arg_constraints = {"locs": constraints.real, "coord_scale": constraints.positive,
                       "component_logits": constraints.real}

    def __init__(self, locs, coord_scale, component_logits):
        self.batch_mode = (locs.dim() > 2)
        assert(self.batch_mode or locs.dim() == 2), \
            "The locs parameter in MixtureOfDiagNormals should be K x D dimensional (or ... x B x K x D in batch mode)"
        if not self.batch_mode:
            assert(coord_scale.dim() == 1), "The coord_scale parameter in MixtureOfDiagNormals should be D dimensional"
            assert(component_logits.dim() == 1), \
                "The component_logits parameter in MixtureOfDiagNormals should be K dimensional"
            assert(component_logits.size(0) == locs.size(0))
            batch_shape = ()
        else:
            assert(coord_scale.dim() > 1), \
                "The coord_scale parameter in MixtureOfDiagNormals should be ... x B x D dimensional"
            assert(component_logits.dim() > 1), \
                "The component_logits parameter in MixtureOfDiagNormals should be ... x B x K dimensional"
            assert(component_logits.size(-1) == locs.size(-2))
            batch_shape = tuple(locs.shape[:-2])
        self.locs = locs
        self.coord_scale = coord_scale
        self.component_logits = component_logits
        self.dim = locs.size(-1)
        if self.dim < 2:
            raise NotImplementedError('This distribution does not support D = 1')
        self.categorical = Categorical(logits=component_logits)
        self.probs = self.categorical.probs
        super().__init__(batch_shape=batch_shape, event_shape=(self.dim,))

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MixtureOfDiagNormalsSharedCovariance, _instance)
        new.batch_mode = True
        batch_shape = torch.Size(batch_shape)
        new.dim = self.dim
        new.locs = self.locs.expand(batch_shape + self.locs.shape[-2:])
        coord_scale_shape = -1 if self.batch_mode else -2
        new.coord_scale = self.coord_scale.expand(batch_shape + self.coord_scale.shape[coord_scale_shape:])
        new.component_logits = self.component_logits.expand(batch_shape + self.component_logits.shape[-1:])
        new.categorical = self.categorical.expand(batch_shape)
        new.probs = self.probs.expand(batch_shape + self.probs.shape[-1:])
        super(MixtureOfDiagNormalsSharedCovariance, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        coord_scale = self.coord_scale.unsqueeze(-2) if self.batch_mode else self.coord_scale
        epsilon = (value.unsqueeze(-2) - self.locs) / coord_scale  # L B K D
        eps_sqr = 0.5 * torch.pow(epsilon, 2.0).sum(-1)  # L B K
        eps_sqr_min = torch.min(eps_sqr, -1)[0]  # L B
        result = self.categorical.logits + (-eps_sqr + eps_sqr_min.unsqueeze(-1))  # L B K
        result = torch.logsumexp(result, dim=-1)  # L B
        result = result - (0.5 * math.log(2.0 * math.pi) * float(self.dim))
        result = result - (torch.log(self.coord_scale).sum(-1))
        result = result - eps_sqr_min
        return result

    def rsample(self, sample_shape=torch.Size()):
        which = self.categorical.sample(sample_shape)
        return _MixDiagNormalSharedCovarianceSample.apply(self.locs, self.coord_scale, self.component_logits,
                                                          self.probs, which, sample_shape + self.coord_scale.shape)


class _MixDiagNormalSharedCovarianceSample(Function):
    @staticmethod
    def forward(ctx, locs, coord_scale, component_logits, pis, which, noise_shape):
        dim = coord_scale.size(-1)
        white = torch.randn(noise_shape, dtype=locs.dtype, device=locs.device)
        n_unsqueezes = locs.dim() - which.dim()
        for _ in range(n_unsqueezes):
            which = which.unsqueeze(-1)
        expand_tuple = tuple(which.shape[:-1] + (dim,))
        loc = torch.gather(locs, -2, which.expand(expand_tuple)).squeeze(-2)
        z = loc + coord_scale * white
        ctx.save_for_backward(z, coord_scale, locs, component_logits, pis)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):

        z, coord_scale, locs, component_logits, pis = ctx.saved_tensors
        K = component_logits.size(-1)
        batch_dims = coord_scale.dim() - 1
        g = grad_output  # l b i

        z_tilde = z / coord_scale  # l b i
        locs_tilde = locs / coord_scale.unsqueeze(-2)  # b j i
        mu_ab = locs_tilde.unsqueeze(-2) - locs_tilde.unsqueeze(-3)  # b k j i
        mu_ab_norm = torch.pow(mu_ab, 2.0).sum(-1).sqrt()  # b k j
        mu_ab /= mu_ab_norm.unsqueeze(-1)  # b k j i
        diagonals = torch.empty((K,), dtype=torch.long, device=z.device)
        torch.arange(K, out=diagonals)
        mu_ab[..., diagonals, diagonals, :] = 0.0

        mu_ll_ab = (locs_tilde.unsqueeze(-2) * mu_ab).sum(-1)  # b k j
        z_ll_ab = (z_tilde.unsqueeze(-2).unsqueeze(-2) * mu_ab).sum(-1)  # l b k j
        z_perp_ab = z_tilde.unsqueeze(-2).unsqueeze(-2) - z_ll_ab.unsqueeze(-1) * mu_ab  # l b k j i
        z_perp_ab_sqr = torch.pow(z_perp_ab, 2.0).sum(-1)  # l b k j

        epsilons = z_tilde.unsqueeze(-2) - locs_tilde  # l b j i
        log_qs = -0.5 * torch.pow(epsilons, 2.0)   # l b j i
        log_q_j = log_qs.sum(-1, keepdim=True)     # l b j 1
        log_q_j_max = torch.max(log_q_j, -2, keepdim=True)[0]
        q_j_prime = torch.exp(log_q_j - log_q_j_max)  # l b j 1
        q_j = torch.exp(log_q_j)  # l b j 1

        q_tot = (pis.unsqueeze(-1) * q_j).sum(-2)  # l b 1
        q_tot_prime = (pis.unsqueeze(-1) * q_j_prime).sum(-2).unsqueeze(-1)  # l b 1 1

        root_two = math.sqrt(2.0)
        mu_ll_ba = torch.transpose(mu_ll_ab, -1, -2)
        logits_grad = torch.erf((z_ll_ab - mu_ll_ab) / root_two) - torch.erf((z_ll_ab + mu_ll_ba) / root_two)
        logits_grad *= torch.exp(-0.5 * z_perp_ab_sqr)  # l b k j

        #                 bi      lbi                               bkji
        mu_ab_sigma_g = ((coord_scale * g).unsqueeze(-2).unsqueeze(-2) * mu_ab).sum(-1)  # l b k j
        logits_grad *= -mu_ab_sigma_g * pis.unsqueeze(-2)  # l b k j
        logits_grad = pis * sum_leftmost(logits_grad.sum(-1) / q_tot, -(1 + batch_dims))  # b k
        logits_grad *= math.sqrt(0.5 * math.pi)

        #           b j                 l b j 1   l b i             l b 1 1
        prefactor = pis.unsqueeze(-1) * q_j_prime * g.unsqueeze(-2) / q_tot_prime  # l b j i
        locs_grad = sum_leftmost(prefactor, -(2 + batch_dims))  # b j i
        coord_scale_grad = sum_leftmost(prefactor * epsilons, -(2 + batch_dims)).sum(-2)  # b i

        return locs_grad, coord_scale_grad, logits_grad, None, None, None
