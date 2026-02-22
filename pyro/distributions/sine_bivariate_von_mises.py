# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from math import pi

import torch
from torch.distributions import VonMises
from torch.distributions.utils import broadcast_all, lazy_property

from pyro.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.ops.special import log_I1


class SineBivariateVonMises(TorchDistribution):
    r"""Unimodal distribution of two dependent angles on the 2-torus (S^1 ⨂ S^1) given by

    .. math::

        C^{-1}\exp(\kappa_1\cos(x-\mu_1) + \kappa_2\cos(x_2 -\mu_2) + \rho\sin(x_1 - \mu_1)\sin(x_2 - \mu_2))

    and

    .. math::

        C = (2\pi)^2 \sum_{i=0} {2i \choose i}
        \left(\frac{\rho^2}{4\kappa_1\kappa_2}\right)^i I_i(\kappa_1)I_i(\kappa_2),

    where I_i(\cdot) is the modified bessel function of first kind, mu's are the locations of the distribution,
    kappa's are the concentration and rho gives the correlation between angles x_1 and x_2.

    This distribution is a submodel of the Bivariate von Mises distribution, called the Sine Distribution [2] in
    directional statistics.

    This distribution is helpful for modeling coupled angles such as torsion angles in peptide chains.
    To infer parameters, use :class:`~pyro.infer.NUTS` or :class:`~pyro.infer.HMC` with priors that
    avoid parameterizations where the distribution becomes bimodal; see note below.

    .. note:: Sample efficiency drops as

        .. math::

            \frac{\rho^2}{\kappa_1\kappa_2} \rightarrow 1

        because the distribution becomes increasingly bimodal. To avoid inefficient sampling use the
        `weighted_correlation` parameter with a skew away from one (e.g.,
        `TransformedDistribution(Beta(5,5), AffineTransform(loc=-1, scale=2))`). The `weighted_correlation`
        should be in [-1,1].

    .. note:: The correlation and weighted_correlation params are mutually exclusive.

    .. note:: In the context of :class:`~pyro.infer.SVI`, this distribution can be used as a likelihood but not for
        latent variables.

    .. note:: Normalization remains accurate up to concentrations of 10,000.

    ** References: **
      1. Probabilistic model for two dependent circular variables Singh, H., Hnizdo, V., and Demchuck, E. (2002)
      2. Protein Bioinformatics and Mixtures of Bivariate von Mises Distributions for Angular Data,
         Mardia, K. V, Taylor, T. C., and Subramaniam, G. (2007)

    :param torch.Tensor phi_loc: location of first angle
    :param torch.Tensor psi_loc: location of second angle
    :param torch.Tensor phi_concentration: concentration of first angle
    :param torch.Tensor psi_concentration: concentration of second angle
    :param torch.Tensor correlation: correlation between the two angles
    :param torch.Tensor weighted_correlation: set correlation to weighted_corr * sqrt(phi_conc*psi_conc)
        to avoid bimodality (see note). The `weighted_correlation` should be in [-1,1].
    """

    arg_constraints = {
        "phi_loc": constraints.real,
        "psi_loc": constraints.real,
        "phi_concentration": constraints.positive,
        "psi_concentration": constraints.positive,
        "correlation": constraints.real,
    }
    support = constraints.independent(constraints.real, 1)
    max_sample_iter = 1000

    def __init__(
        self,
        phi_loc,
        psi_loc,
        phi_concentration,
        psi_concentration,
        correlation=None,
        weighted_correlation=None,
        validate_args=None,
    ):
        assert (correlation is None) != (weighted_correlation is None)

        if weighted_correlation is not None:
            sqrt_ = (
                torch.sqrt if isinstance(phi_concentration, torch.Tensor) else math.sqrt
            )
            correlation = weighted_correlation * sqrt_(
                phi_concentration * psi_concentration
            )

        (
            phi_loc,
            psi_loc,
            phi_concentration,
            psi_concentration,
            correlation,
        ) = broadcast_all(
            phi_loc, psi_loc, phi_concentration, psi_concentration, correlation
        )

        max_conc = torch.maximum(
            torch.max(phi_concentration), torch.max(psi_concentration)
        )
        assrt_hstr = (
            "Normalization of SineBiviateVonMises is inaccurate for"
            f"current max concentration ({max_conc} > 10,000)."
        )
        assert max_conc <= torch.tensor(10_000.0), assrt_hstr

        self.phi_loc = phi_loc
        self.psi_loc = psi_loc
        self.phi_concentration = phi_concentration
        self.psi_concentration = psi_concentration
        self.correlation = correlation
        event_shape = torch.Size([2])
        batch_shape = phi_loc.shape

        super().__init__(batch_shape, event_shape, validate_args)

        if self._validate_args and torch.any(
            phi_concentration * psi_concentration <= correlation**2
        ):
            warnings.warn(
                f"{self.__class__.__name__} bimodal due to concentration-correlation relation, "
                f"sampling will likely fail.",
                UserWarning,
            )

    @lazy_property
    def norm_const(self):
        corr = self.correlation.view(1, -1)
        conc = torch.stack(
            (self.phi_concentration, self.psi_concentration), dim=-1
        ).view(-1, 2)
        m = torch.arange(50, device=self.phi_loc.device).view(-1, 1)
        tiny = torch.finfo(corr.dtype).tiny
        fs = (
            SineBivariateVonMises._lbinoms(m.max() + 1).view(-1, 1)
            + m * torch.log((corr**2).clamp(min=tiny))
            - m * torch.log(4 * torch.prod(conc, dim=-1))
        )
        num_I1terms = torch.maximum(
            torch.tensor(501),
            torch.max(self.phi_concentration) + torch.max(self.psi_concentration),
        ).int()

        fs += log_I1(m.max(), conc, num_I1terms).sum(-1)

        mfs = fs.max()
        norm_const = 2 * torch.log(torch.tensor(2 * pi)) + mfs + (fs - mfs).logsumexp(0)
        return norm_const.reshape(self.phi_loc.shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indv = self.phi_concentration * torch.cos(
            value[..., 0] - self.phi_loc
        ) + self.psi_concentration * torch.cos(value[..., 1] - self.psi_loc)
        corr = (
            self.correlation
            * torch.sin(value[..., 0] - self.phi_loc)
            * torch.sin(value[..., 1] - self.psi_loc)
        )
        return indv + corr - self.norm_const

    def sample(self, sample_shape=torch.Size()):
        """
        ** References: **
            1. A New Unified Approach for the Simulation of aWide Class of Directional Distributions
               John T. Kent, Asaad M. Ganeiber & Kanti V. Mardia (2018)
        """
        assert not torch._C._get_tracing_state(), "jit not supported"
        sample_shape = torch.Size(sample_shape)

        corr = self.correlation
        conc = torch.stack((self.phi_concentration, self.psi_concentration))

        eig = 0.5 * (conc[0] - corr**2 / conc[1])
        eig = torch.stack((torch.zeros_like(eig), eig))
        eigmin = torch.where(
            eig[1] < 0, eig[1], torch.zeros_like(eig[1], dtype=eig.dtype)
        )
        eig = eig - eigmin
        b0 = self._bfind(eig)

        total = sample_shape.numel()
        missing = total * torch.ones(
            (self.batch_shape.numel(),), dtype=torch.int, device=conc.device
        )
        start = torch.zeros_like(missing, device=conc.device)
        phi = torch.empty(
            (2, *missing.shape, total), dtype=corr.dtype, device=conc.device
        )

        max_iter = SineBivariateVonMises.max_sample_iter

        # flatten batch_shape
        conc = conc.view(2, -1, 1)
        eigmin = eigmin.view(-1, 1)
        corr = corr.reshape(-1, 1)
        eig = eig.view(2, -1)
        b0 = b0.view(-1)
        phi_den = log_I1(0, conc[1]).view(-1, 1)
        lengths = torch.arange(total, device=conc.device).view(1, -1)

        while torch.any(missing > 0) and max_iter:
            curr_conc = conc[:, missing > 0, :]
            curr_corr = corr[missing > 0]
            curr_eig = eig[:, missing > 0]
            curr_b0 = b0[missing > 0]

            x = (
                torch.distributions.Normal(0.0, torch.rsqrt(1 + 2 * curr_eig / curr_b0))
                .sample((missing[missing > 0].min(),))
                .view(2, -1, missing[missing > 0].min())
            )
            x /= x.norm(dim=0)[None, ...]  # Angular Central Gaussian distribution

            lf = (
                curr_conc[0] * (x[0] - 1)
                + eigmin[missing > 0]
                + log_I1(
                    0, torch.sqrt(curr_conc[1] ** 2 + (curr_corr * x[1]) ** 2)
                ).squeeze(0)
                - phi_den[missing > 0]
            )
            assert lf.shape == ((missing > 0).sum(), missing[missing > 0].min())

            lg_inv = (
                1.0
                - curr_b0.view(-1, 1) / 2
                + torch.log(
                    curr_b0.view(-1, 1) / 2 + (curr_eig.view(2, -1, 1) * x**2).sum(0)
                )
            )
            assert lg_inv.shape == lf.shape

            accepted = (
                torch.distributions.Uniform(
                    0.0, torch.ones((), device=conc.device)
                ).sample(lf.shape)
                < (lf + lg_inv).exp()
            )

            phi_mask = torch.zeros(
                (*missing.shape, total), dtype=torch.bool, device=conc.device
            )
            phi_mask[missing > 0] = torch.logical_and(
                lengths < (start[missing > 0] + accepted.sum(-1)).view(-1, 1),
                lengths >= start[missing > 0].view(-1, 1),
            )

            phi[:, phi_mask] = x[:, accepted]

            start[missing > 0] += accepted.sum(-1)
            missing[missing > 0] -= accepted.sum(-1)
            max_iter -= 1

        if max_iter == 0 or torch.any(missing > 0):
            raise ValueError(
                "maximum number of iterations exceeded; "
                "try increasing `SineBivariateVonMises.max_sample_iter`"
            )

        phi = torch.atan2(phi[1], phi[0])

        alpha = torch.sqrt(conc[1] ** 2 + (corr * torch.sin(phi)) ** 2)
        beta = torch.atan(corr / conc[1] * torch.sin(phi))

        psi = VonMises(beta, alpha).sample()

        phi_psi = torch.stack(
            (
                (phi + self.phi_loc.reshape((-1, 1)) + pi) % (2 * pi) - pi,
                (psi + self.psi_loc.reshape((-1, 1)) + pi) % (2 * pi) - pi,
            ),
            dim=-1,
        ).permute(1, 0, 2)
        return phi_psi.reshape(*sample_shape, *self.batch_shape, *self.event_shape)

    @property
    def mean(self):
        return torch.stack((self.phi_loc, self.psi_loc), dim=-1)

    @classmethod
    def infer_shapes(cls, **arg_shapes):
        batch_shape = torch.Size(broadcast_shape(*arg_shapes.values()))
        return batch_shape, torch.Size([2])

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SineBivariateVonMises, _instance)
        batch_shape = torch.Size(batch_shape)
        for k in SineBivariateVonMises.arg_constraints.keys():
            setattr(new, k, getattr(self, k).expand(batch_shape))
        new.norm_const = self.norm_const.expand(batch_shape)
        super(SineBivariateVonMises, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _bfind(self, eig):
        b = (
            eig.shape[0]
            / 2
            * torch.ones(self.batch_shape, dtype=eig.dtype, device=eig.device)
        )
        g1 = torch.sum(1 / (b + 2 * eig) ** 2, dim=0)
        g2 = torch.sum(-2 / (b + 2 * eig) ** 3, dim=0)
        return torch.where(eig.norm(0) != 0, b - g1 / g2, b)

    @staticmethod
    def _lbinoms(n):
        ns = torch.arange(n, device=n.device)
        num = torch.lgamma(2 * ns + 1)
        den = torch.lgamma(ns + 1)
        return num - 2 * den
