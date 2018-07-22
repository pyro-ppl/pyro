from __future__ import absolute_import, division, print_function
import math

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints, Categorical

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import sum_leftmost


class GaussianScaleMixture(TorchDistribution):
    """
    Mixture of Normal distributions with zero mean and diagonal covariance matrices.

    That is, this distribution is a mixture with K components, where each
    component distribution is a D-dimensional Normal distribution with zero
    mean and a D-dimensional diagonal covariance matrix. The K different
    covariance matrices are controlled by the parameters `scale` and `lambdas`.
    That is, the covariance matrix of the k'th component is given by

    Sigma_ii = (lambda_k * scale_i) ** 2

    where `lambda`_k is a positive scale factor and `scale`_i are positive
    scale parameters shared between all K components. The mixture weights are
    controlled by a K-dimensional vector of softmax logits, `logits`. This
    distribution implements pathwise derivatives for samples from the distribution.
    This distribution does not currently support batched parameters.

    See reference [1] for details on the implementations of the pathwise
    derivative. Please consider citing this reference if you use the pathwise
    derivative in your research.

    [1] Pathwise Derivatives for Multivariate Distributions, Martin Jankowiak &
    Theofanis Karaletsos. arXiv:1806.01856

    Note that this distribution supports both even and odd dimensions, but the
    former should be more a bit higher precision, since it doesn't use any erfs in
    the backward call.

    :param torch.tensor scale: D-dimensional vector of scale
    :param torch.tensor logits: K-dimensional vector of logits
    :param torch.tensor lambdas: K-dimensional matrix of sigma multipliers
    """
    has_rsample = True
    arg_constraints = {"lambdas": constraints.positive, "scale": constraints.positive,
                       "logits": constraints.real}

    def __init__(self, scale, logits, lambdas):
        self.dim = scale.size(0)
        assert(scale.dim() == 1), "The scale parameter in GaussianScaleMixture should be D dimensional"
        assert(lambdas.dim() == 1), "The lambdas parameter in GaussianScaleMixture should be K dimensional"
        assert(logits.dim() == 1), "The logits parameter in GaussianScaleMixture should be K dimensional"
        assert(logits.shape == lambdas.shape), \
            "The logits and lambdas parameters in GaussianScaleMixture should be K dimensional"
        self.scale = scale
        self.logits = logits
        self.lambdas = lambdas
        self.coeffs = self._compute_coeffs()
        self.categorical = Categorical(logits=logits)
        super(GaussianScaleMixture, self).__init__()

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
        epsilon_sqr = torch.pow(value / self.scale, 2.0).sum()
        lambdas_power = torch.pow(self.lambdas, -self.dim)
        result = lambdas_power * self.categorical.probs * \
            torch.exp(-0.5 * epsilon_sqr / torch.pow(self.lambdas, 2.0))  # K
        result = torch.log(result.sum())
        result -= 0.5 * math.log(2.0 * math.pi) * float(self.dim)
        result -= torch.log(self.scale).sum()
        return result

    def rsample(self, sample_shape=torch.Size()):
        which = self.categorical.sample(sample_shape)
        return _GSMSample.apply(self.scale, self.logits, self.lambdas, self.categorical.probs, which,
                                sample_shape + torch.Size((self.dim,)), self.coeffs)


class _GSMSample(Function):
    @staticmethod
    def forward(ctx, scale, logits, lambdas, pis, which, shape, coeffs):
        white = scale.new(shape).normal_()
        which_lambdas = lambdas[which].unsqueeze(-1)
        z = scale * which_lambdas * white
        ctx.save_for_backward(z, scale, logits, lambdas, pis, coeffs)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        z, scale, logits, lambdas, pis, coeffs = ctx.saved_tensors
        dim = scale.size(0)
        g = grad_output  # l i
        g = g.unsqueeze(-2)  # l 1 i

        lambdas_sqr = torch.pow(lambdas, 2.0)  # j
        epsilons = z / scale  # l i
        epsilons_sqr = torch.pow(epsilons, 2.0)  # l i
        r_sqr = epsilons_sqr.sum(-1)  # l
        r_sqr_j = r_sqr.unsqueeze(-1) / lambdas_sqr  # l j
        log_scale = torch.log(scale)  # i
        scale_product = log_scale.sum().exp()
        lambdas_power = torch.pow(lambdas, float(dim))

        q_j = torch.exp(-0.5 * r_sqr_j) / math.power(2.0 * math.pi, 0.5 * float(dim))  # l j
        q_j /= scale_product * lambdas_power  # l j
        q_tot = (pis * q_j).sum(-1)  # l
        q_tot = q_tot.unsqueeze(-1)  # l 1

        Phi_j = torch.exp(-0.5 * r_sqr_j)  # l j
        exponents = - torch.arange(1, int(dim/2) + 1, 1)
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
        logits_grad = sum_leftmost(logits_grad, -1) * math.power(2.0 * math.pi, -0.5 * float(dim))
        logits_grad = pis * logits_grad / (lambdas_power * scale_product)
        logits_grad = logits_grad - logits_grad.sum() * pis

        prefactor = pis.unsqueeze(-1) * q_j.unsqueeze(-1) * g / q_tot.unsqueeze(-1)  # l j i
        scale_grad = sum_leftmost(prefactor * epsilons.unsqueeze(-2), -1)
        lambdas_grad = sum_leftmost((prefactor * z.unsqueeze(-2)).sum(-1) / lambdas, -1)

        return scale_grad, logits_grad, lambdas_grad, None, None, None, None
