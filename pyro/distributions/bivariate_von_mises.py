# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings
from copy import copy
from functools import reduce
from math import pi

import torch
from torch.autograd import grad
from torch.distributions import VonMises
from torch.distributions.utils import broadcast_all

from pyro.distributions import TorchDistribution, constraints


def _flat_dim(size):
    return torch.prod(torch.tensor(size), dim=0)


def log_I1(orders: int, value: torch.Tensor, terms=250):
    """ Compute first n log modified bessel function of first kind
        log(I_v(z)) = v*log(z/2) + log(sum_{k=0}^inf exp[2*k*log(z/2) - sum_kk^k log(kk) - lgamma(v + k + 1)])

        :param orders: orders of the log modified bessel function.
        :param value: values to compute modified bessel function for
        :param terms: truncation of summation
        :return: 0 to orders modified bessel function [shape:(orders, value.squeeze().shape)]
    """
    orders = orders + 1
    if len(value.size()) == 0:
        vshape = torch.Size([1])
    else:
        vshape = copy(value.shape)
    value = value.reshape(-1, 1)

    k = torch.arange(terms)
    lgammas_all = torch.lgamma(torch.arange(1, terms + orders + 1))
    assert lgammas_all.shape == (orders + terms,)  # lgamma(0) = inf => start from 1

    lvalues = torch.log(value / 2) * k.view(1, -1)
    assert lvalues.shape == (_flat_dim(vshape), terms)

    lfactorials = lgammas_all[:terms]
    assert lfactorials.shape == (terms,)

    lgammas = lgammas_all.repeat(orders).view(orders, -1)
    assert lgammas.shape == (orders, terms + orders)  # lgamma(0) = inf => start from 1

    indices = k[:orders].view(-1, 1) + k.view(1, -1)
    assert indices.shape == (orders, terms)

    seqs = (2 * lvalues[None, :, :] - lfactorials[None, None, :] - lgammas.gather(1, indices)[:, None, :]).logsumexp(-1)
    assert seqs.shape == (orders, _flat_dim(vshape))

    i1s = lvalues[..., :orders].T + seqs
    assert i1s.shape == (orders, _flat_dim(vshape))
    return i1s.view(-1, *vshape)


class SineBivariateVonMises(TorchDistribution):
    """ Distribution of two dependent angles on the 2-torus (S^1 ⨂ S^1) given by
    .. math::
      C^{-1}\\exp(\\kappa_1\\cos(x-\\mu_1) + \\kappa_2\\cos(x_2 -\\mu_2) + \\rho\\sin(x_1 - \\mu_1)\\sin(x_2 - \\mu_2))
    and
    .. math::
      C = (2\\pi)^2 \\sum_{i=0} {2i \\choose i}
        \\left(\\frac{\\rho^2}{4\\kappa_1\\kappa_2}\\right)^2 I_i(\\kappa_1)I_i(\\kappa_2),
    where I_i(\\cdot) is the modified bessel function of first kind, \\mu 's are the locations of the distribution,
    \\kappa 's are the concentration and \\rho correlation between angles x_1 and x_2.

    Note: Sample efficiency drops as
    .. math::
        \\frac{\\rho}{\\kappa_1\\kappa_2} \\rightarrow 1
    and the distribution becomes bimodal.

    ** References: **
      1. Probabilistic model for two dependent circular variables
         Singh, H., Hnizdo, V., and Demchuck, E. (2002)

    :param torch.Tensor phi_loc: location of first angle
    :param torch.Tensor psi_loc: location of second angle
    :param torch.Tensor phi_concentration: concentration of first angle
    :param torch.Tensor psi_concentration: concentration of second angle
    :param torch.Tensor correlation: correlation between the two angles
    :param torch.Tensor weighted_correlation: set correlation to
    .. math:
        w_\\rho\\sqrt{\\kappa_1\\kappa_2},
    to avoid bimodality (see note).
    """

    arg_constraints = {'phi_loc': constraints.real, 'psi_loc': constraints.real,
                       'phi_concentration': constraints.positive, 'psi_concentration': constraints.positive,
                       'correlation': constraints.real}
    support = constraints.real
    max_sample_iter = 1000

    def __init__(self, phi_loc, psi_loc, phi_concentration, psi_concentration, correlation=None,
                 weighted_correlation=None, validate_args=None):

        assert (correlation is None) != (weighted_correlation is None)

        if weighted_correlation is not None:
            correlation = weighted_correlation * (phi_concentration * psi_concentration).sqrt() + 1e-8

        if torch.any(phi_concentration * psi_concentration <= correlation ** 2):
            warnings.warn(
                f'{self.__class__.__name__} bimodal due to concentration-correlation relation, '
                f'sampling will likely fail.', UserWarning)

        phi_loc, psi_loc, phi_concentration, psi_concentration, correlation = broadcast_all(phi_loc, psi_loc,
                                                                                            phi_concentration,
                                                                                            psi_concentration,
                                                                                            correlation)
        self.phi_loc = phi_loc
        self.psi_loc = psi_loc
        self.phi_concentration = phi_concentration
        self.psi_concentration = psi_concentration
        self.correlation = correlation
        self._norm_const = None
        event_shape = torch.Size([2])
        batch_shape = phi_loc.shape
        super(SineBivariateVonMises, self).__init__(batch_shape, event_shape, validate_args)

    @property
    def norm_const(self):
        if hasattr(self, '_norm_const') and self._norm_const is None:
            corr = self.correlation + 1e-8
            conc = torch.hstack((self.phi_concentration, self.psi_concentration))
            m = torch.arange(50)
            fs = SineBivariateVonMises._lbinoms(m.max() + 1) + 2 * m * torch.log(corr) - m * torch.log(
                4 * torch.prod(conc, dim=0))
            fs += torch.sum(log_I1(m.max(), conc), dim=-1)
            mfs = fs.max()
            self._norm_const = 2 * torch.log(torch.tensor(2 * pi)) + mfs + (fs - mfs).logsumexp(-1)
        return self._norm_const

    def log_prob(self, value):
        if len(value.size()) == 1:
            value = value[None, :]
        indv = self.phi_concentration * torch.cos(value[..., 0] - self.phi_loc) + self.psi_concentration * torch.cos(
            value[..., 1] - self.psi_loc)
        corr = self.correlation * torch.sin(value[..., 0] - self.phi_loc) * torch.sin(value[..., 1] - self.psi_loc)
        return indv + corr - self.norm_const

    def sample(self, sample_shape=torch.Size()):
        """ ** References: **
              1. A New Unified Approach for the Simulation of aWide Class of Directional Distributions
                John T. Kent, Asaad M. Ganeiber & Kanti V. Mardia (2018)
        """

        corr = self.correlation
        conc = torch.stack((self.phi_concentration, self.psi_concentration))

        eig = 0.5 * (conc[0] - corr ** 2 / conc[1])
        eig = torch.stack((torch.zeros_like(eig), eig))
        eigmin = torch.where(eig[1] < 0, eig[1], torch.zeros_like(eig[1], dtype=eig.dtype))
        eig = eig - eigmin
        eig.requires_grad = True
        b0 = self._bfind(eig)

        total = int(torch.prod(torch.tensor(sample_shape)))
        missing = total * torch.ones(reduce(lambda a, b: a * b, self.batch_shape, 1), dtype=torch.int)
        start = torch.zeros_like(missing)
        phi = torch.empty((2, *missing.shape, total), dtype=corr.dtype)

        max_iter = SineBivariateVonMises.max_sample_iter

        # flatten batch_shape
        conc = conc.view(2, -1, 1)
        eigmin = eigmin.view(-1, 1)
        corr = corr.view(-1, 1)
        eig = eig.view(2, -1)
        b0 = b0.view(-1)
        phi_den = log_I1(0, conc[1]).view(-1, 1)

        while torch.any(missing > 0) and max_iter:
            curr_conc = conc[:, missing > 0, :]
            curr_corr = corr[missing > 0]
            curr_eig = eig[:, missing > 0]
            curr_b0 = b0[missing > 0]

            x = torch.distributions.Normal(0., torch.sqrt(1 + 2 * curr_eig / curr_b0)).sample(
                (missing[missing > 0].min(),)).view(2, -1, missing[missing > 0].min())
            x /= x.norm(dim=0)[None, ...]  # Angular Central Gaussian distribution

            lf = curr_conc[0] * (x[0] - 1) + eigmin[missing > 0] + log_I1(0, torch.sqrt(
                curr_conc[1] ** 2 + (curr_corr * x[1]) ** 2)).squeeze(0) - phi_den[missing > 0]
            assert lf.shape == ((missing > 0).sum(), missing[missing > 0].min())

            lg_inv = 1. - curr_b0.view(-1, 1) / 2 + torch.log(
                curr_b0.view(-1, 1) / 2 + (curr_eig.view(2, -1, 1) * x ** 2).sum(0))
            assert lg_inv.shape == lf.shape

            accepted = torch.distributions.Uniform(0., 1.).sample(lf.shape) < (lf + lg_inv).exp()

            phi_mask = torch.zeros((*missing.shape, total), dtype=torch.bool)
            phi_mask[missing > 0] = torch.logical_and(
                torch.arange(total).view(1, -1) < (start[missing > 0] + accepted.sum(-1)).view(-1, 1),
                torch.arange(total).view(1, -1) >= start[missing > 0].view(-1, 1))

            phi[:, phi_mask] = x[:, accepted]

            start[missing > 0] += accepted.sum(-1)
            missing[missing > 0] -= accepted.sum(-1)
            max_iter -= 1

        if max_iter == 0 or torch.any(missing > 0):
            raise Exception("sample error")  # FIXME: what would be an appropriate exception?

        phi = torch.atan2(phi[1], phi[0])

        alpha = torch.sqrt(conc[1] ** 2 + (corr * torch.sin(phi)) ** 2)
        beta = torch.atan(corr / conc[1] * torch.sin(phi))

        psi = VonMises(beta, alpha).sample()
        phi = phi.view((*self.batch_shape, total))
        psi = psi.view((*self.batch_shape, total))

        phi_psi = torch.vstack(((phi + self.phi_loc.view((*self.batch_shape, 1)) + pi) % (2 * pi) - pi,
                                (psi + self.psi_loc.view((*self.batch_shape, 1)) + pi) % (2 * pi) - pi)).T
        return phi_psi.view(*sample_shape, *self.batch_shape, *self.event_shape)

    @property
    def mean(self):
        return torch.stack((self.phi_loc, self.psi_loc))

    def expand(self, batch_shape, _instance=None):
        try:
            return super(SineBivariateVonMises, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            phi_loc = self.phi_loc.expand(batch_shape)
            psi_loc = self.psi_loc.expand(batch_shape)
            phi_conc = self.phi_concentration.expand(batch_shape)
            psi_conc = self.psi_concentration.expand(batch_shape)
            corr = self.correlation.expand(batch_shape)

            return type(self)(phi_loc, psi_loc, phi_conc, psi_conc, corr, validate_args=validate_args)

    def _bfind(self, eig):
        eig = eig.view(2, -1)
        b = eig.shape[0] / 2 * torch.ones(self.batch_shape, requires_grad=True, dtype=torch.float)
        b = b.ravel()
        for i in torch.arange(b.shape[0])[eig.norm(0) != 0].T:
            curr_eig = eig[:, i]  # make indexed a non-leaf
            curr_b = b[i]  # make indexed a non-leaf
            g1 = grad(1 - torch.sum(1 / (curr_b + 2 * curr_eig)), curr_b, create_graph=True)[0]
            g2 = grad(g1, curr_b)[0]
            b[i] -= (1 / g2) * g1
        return b.view(self.batch_shape)

    @staticmethod
    def _lbinoms(n):
        num = torch.lgamma(2 * torch.arange(n) + 1)
        den = torch.lgamma(torch.arange(n) + 1)
        return num - 2 * den
