# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import Categorical, constraints

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import sum_leftmost


class GaussianScaleMixture(TorchDistribution):
    """
    Mixture of Normal distributions with zero mean and diagonal covariance
    matrices.

    That is, this distribution is a mixture with K components, where each
    component distribution is a D-dimensional Normal distribution with zero
    mean and a D-dimensional diagonal covariance matrix. The K different
    covariance matrices are controlled by the parameters `coord_scale` and
    `component_scale`.  That is, the covariance matrix of the k'th component is
    given by

    Sigma_ii = (component_scale_k * coord_scale_i) ** 2   (i = 1, ..., D)

    where `component_scale_k` is a positive scale factor and `coord_scale_i`
    are positive scale parameters shared between all K components. The mixture
    weights are controlled by a K-dimensional vector of softmax logits,
    `component_logits`. This distribution implements pathwise derivatives for
    samples from the distribution. This distribution does not currently
    support batched parameters.

    See reference [1] for details on the implementations of the pathwise
    derivative. Please consider citing this reference if you use the pathwise
    derivative in your research.

    [1] Pathwise Derivatives for Multivariate Distributions, Martin Jankowiak &
    Theofanis Karaletsos. arXiv:1806.01856

    Note that this distribution supports both even and odd dimensions, but the
    former should be more a bit higher precision, since it doesn't use any erfs
    in the backward call. Also note that this distribution does not support
    D = 1.

    :param torch.tensor coord_scale: D-dimensional vector of scales
    :param torch.tensor component_logits: K-dimensional vector of logits
    :param torch.tensor component_scale: K-dimensional vector of scale multipliers
    """
    has_rsample = True
    arg_constraints = {"component_scale": constraints.positive, "coord_scale": constraints.positive,
                       "component_logits": constraints.real}

    def __init__(self, coord_scale, component_logits, component_scale):
        self.dim = coord_scale.size(0)
        if self.dim < 2:
            raise NotImplementedError('This distribution does not support D = 1')
        assert(coord_scale.dim() == 1), "The coord_scale parameter in GaussianScaleMixture should be D dimensional"
        assert(component_scale.dim() == 1), \
            "The component_scale parameter in GaussianScaleMixture should be K dimensional"
        assert(component_logits.dim() == 1), \
            "The component_logits parameter in GaussianScaleMixture should be K dimensional"
        assert(component_logits.shape == component_scale.shape), \
            "The component_logits and component_scale parameters in GaussianScaleMixture should be K dimensional"
        self.coord_scale = coord_scale
        self.component_logits = component_logits
        self.component_scale = component_scale
        self.coeffs = self._compute_coeffs()
        self.categorical = Categorical(logits=component_logits)
        super().__init__(event_shape=(self.dim,))

    def _compute_coeffs(self):
        """
        These coefficients are used internally in the backward call.
        """
        dimov2 = int(self.dim / 2)  # this is correct for both even and odd dimensions
        coeffs = torch.ones(dimov2)
        for k in range(dimov2 - 1):
            coeffs[k + 1:] *= self.dim - 2 * (k + 1)
        return coeffs

    def log_prob(self, value):
        assert value.dim() == 1 and value.size(0) == self.dim
        epsilon_sqr = torch.pow(value / self.coord_scale, 2.0).sum()
        component_scale_log_power = self.component_scale.log() * -self.dim
        # logits in Categorical is already normalized
        result = torch.logsumexp(
            component_scale_log_power + self.categorical.logits +
            -0.5 * epsilon_sqr / torch.pow(self.component_scale, 2.0), dim=-1)  # K
        result -= 0.5 * math.log(2.0 * math.pi) * float(self.dim)
        result -= self.coord_scale.log().sum()
        return result

    def rsample(self, sample_shape=torch.Size()):
        which = self.categorical.sample(sample_shape)
        return _GSMSample.apply(self.coord_scale, self.component_logits, self.component_scale, self.categorical.probs,
                                which, sample_shape + torch.Size((self.dim,)), self.coeffs)


class _GSMSample(Function):
    @staticmethod
    def forward(ctx, coord_scale, component_logits, component_scale, pis, which, shape, coeffs):
        white = coord_scale.new(shape).normal_()
        which_component_scale = component_scale[which].unsqueeze(-1)
        z = coord_scale * which_component_scale * white
        ctx.save_for_backward(z, coord_scale, component_logits, component_scale, pis, coeffs)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        z, coord_scale, component_logits, component_scale, pis, coeffs = ctx.saved_tensors
        dim = coord_scale.size(0)
        g = grad_output  # l i
        g = g.unsqueeze(-2)  # l 1 i

        component_scale_sqr = torch.pow(component_scale, 2.0)  # j
        epsilons = z / coord_scale  # l i
        epsilons_sqr = torch.pow(epsilons, 2.0)  # l i
        r_sqr = epsilons_sqr.sum(-1, keepdim=True)  # l
        r_sqr_j = r_sqr / component_scale_sqr  # l j
        coord_scale_product = coord_scale.prod()
        component_scale_power = torch.pow(component_scale, float(dim))

        q_j = torch.exp(-0.5 * r_sqr_j) / math.pow(2.0 * math.pi, 0.5 * float(dim))  # l j
        q_j /= coord_scale_product * component_scale_power  # l j
        q_tot = (pis * q_j).sum(-1, keepdim=True)  # l

        Phi_j = torch.exp(-0.5 * r_sqr_j)  # l j
        exponents = - torch.arange(1., int(dim/2) + 1., 1.)
        if z.dim() > 1:
            r_j_poly = r_sqr_j.unsqueeze(-1).expand(-1, -1, int(dim/2))  # l j d/2
        else:
            r_j_poly = r_sqr_j.unsqueeze(-1).expand(-1, int(dim/2))  # l j d/2
        r_j_poly = coeffs * torch.pow(r_j_poly, exponents)
        Phi_j *= r_j_poly.sum(-1)
        if dim % 2 == 1:
            root_two = math.sqrt(2.0)
            extra_term = coeffs[-1] * math.sqrt(0.5 * math.pi) * (1.0 - torch.erf(r_sqr_j.sqrt() / root_two))  # l j
            Phi_j += extra_term * torch.pow(r_sqr_j, -0.5 * float(dim))

        logits_grad = (z.unsqueeze(-2) * Phi_j.unsqueeze(-1) * g).sum(-1)  # l j
        logits_grad /= q_tot
        logits_grad = sum_leftmost(logits_grad, -1) * math.pow(2.0 * math.pi, -0.5 * float(dim))
        logits_grad = pis * logits_grad / (component_scale_power * coord_scale_product)
        logits_grad = logits_grad - logits_grad.sum() * pis

        prefactor = pis.unsqueeze(-1) * q_j.unsqueeze(-1) * g / q_tot.unsqueeze(-1)  # l j i
        coord_scale_grad = sum_leftmost(prefactor * epsilons.unsqueeze(-2), -1)
        component_scale_grad = sum_leftmost((prefactor * z.unsqueeze(-2)).sum(-1) / component_scale, -1)

        return coord_scale_grad, logits_grad, component_scale_grad, None, None, None, None
