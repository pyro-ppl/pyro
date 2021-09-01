# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings
from math import pi

import torch
from torch import broadcast_shapes
from torch.distributions import Uniform

from pyro.distributions import constraints

from .torch_distribution import TorchDistribution


class SineSkewed(TorchDistribution):
    """Sine Skewing [1] is a procedure for producing a distribution that breaks pointwise symmetry on a torus
    distribution. The new distribution is called the Sine Skewed X distribution, where X is the name of the (symmetric)
    base distribution.

    Torus distributions are distributions with support on products of circles
    (i.e., ⨂^d S^1 where S^1=[-pi,pi) ). So, a 0-torus is a point, the 1-torus is a circle,
    and the 2-torus is commonly associated with the donut shape.

    The Sine Skewed X distribution is parameterized by a weight parameter for each dimension of the event of X.
    For example with a von Mises distribution over a circle (1-torus), the Sine Skewed von Mises Distribution has one
    skew parameter. The skewness parameters can be inferred using :class:`~pyro.infer.HMC` or :class:`~pyro.infer.NUTS`.
    For example, the following will produce a uniform prior over skewness for the 2-torus,::

        def model(obs):
            # Sine priors
            phi_loc = pyro.sample('phi_loc', VonMises(pi, 2.))
            psi_loc = pyro.sample('psi_loc', VonMises(-pi / 2, 2.))
            phi_conc = pyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
            psi_conc = pyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
            corr_scale = pyro.sample('corr_scale', Beta(2., 5.))

            # SS prior
            skew_phi = pyro.sample('skew_phi', Uniform(-1., 1.))
            psi_bound = 1 - skew_phi.abs()
            skew_psi = pyro.sample('skew_psi', Uniform(-1., 1.))
            skewness = torch.stack((skew_phi, psi_bound * skew_psi), dim=-1)
            assert skewness.shape == (num_mix_comp, 2)

            with pyro.plate('obs_plate'):
                sine = SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc,
                                             phi_concentration=1000 * phi_conc,
                                             psi_concentration=1000 * psi_conc,
                                             weighted_correlation=corr_scale)
                return pyro.sample('phi_psi', SineSkewed(sine, skewness), obs=obs)

    To ensure the skewing does not alter the normalization constant of the (Sine Bivaraite von Mises) base
    distribution the skewness parameters are constraint. The constraint requires the sum of the absolute values of
    skewness to be less than or equal to one.
    So for the above snippet it must hold that::

        skew_phi.abs()+skew_psi.abs() <= 1

    We handle this in the prior by computing psi_bound and use it to scale skew_psi.
    We do **not** use psi_bound as::

        skew_psi = pyro.sample('skew_psi', Uniform(-psi_bound, psi_bound))

    as it would make the support for the Uniform distribution dynamic.

    In the context of :class:`~pyro.infer.SVI`, this distribution can freely be used as a likelihood, but use as
    latent variables it will lead to slow inference for 2 and higher dim toruses. This is because the base_dist
    cannot be reparameterized.

    .. note:: An event in the base distribution must be on a d-torus, so the event_shape must be (d,).

    .. note:: For the skewness parameter, it must hold that the sum of the absolute value of its weights for an event
        must be less than or equal to one. See eq. 2.1 in [1].

    ** References: **
      1. Sine-skewed toroidal distributions and their application in protein bioinformatics
         Ameijeiras-Alonso, J., Ley, C. (2019)

    :param torch.distributions.Distribution base_dist: base density on a d-dimensional torus. Supported base
        distributions include: 1D :class:`~pyro.distributions.VonMises`,
        :class:`~pyro.distributions.SineBivariateVonMises`, 1D :class:`~pyro.distributions.ProjectedNormal`, and
        :class:`~pyro.distributions.Uniform` (-pi, pi).
    :param torch.tensor skewness: skewness of the distribution.
    """

    arg_constraints = {
        "skewness": constraints.independent(constraints.interval(-1.0, 1.0), 1)
    }

    support = constraints.independent(constraints.real, 1)

    def __init__(self, base_dist: TorchDistribution, skewness, validate_args=None):
        assert (
            base_dist.event_shape == skewness.shape[-1:]
        ), "Sine Skewing is only valid with a skewness parameter for each dimension of `base_dist.event_shape`."

        if (skewness.abs().sum(-1) > 1.0).any():
            warnings.warn("Total skewness weight shouldn't exceed one.", UserWarning)

        batch_shape = broadcast_shapes(base_dist.batch_shape, skewness.shape[:-1])
        event_shape = skewness.shape[-1:]
        self.skewness = skewness.broadcast_to(batch_shape + event_shape)
        self.base_dist = base_dist.expand(batch_shape)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        if self._validate_args and base_dist.mean.device != skewness.device:
            raise ValueError(
                f"base_density: {base_dist.__class__.__name__} and SineSkewed "
                f"must be on same device."
            )

    def __repr__(self):
        args_string = ", ".join(
            [
                "{}: {}".format(
                    p,
                    getattr(self, p)
                    if getattr(self, p).numel() == 1
                    else getattr(self, p).size(),
                )
                for p in self.arg_constraints.keys()
            ]
        )
        return (
            self.__class__.__name__
            + "("
            + f"base_density: {str(self.base_dist)}, "
            + args_string
            + ")"
        )

    def sample(self, sample_shape=torch.Size()):
        bd = self.base_dist
        ys = bd.sample(sample_shape)
        u = Uniform(0.0, self.skewness.new_ones(())).sample(
            sample_shape + self.batch_shape
        )

        # Section 2.3 step 3 in [1]
        mask = u <= 0.5 + 0.5 * (
            self.skewness * torch.sin((ys - bd.mean) % (2 * pi))
        ).sum(-1)
        mask = mask[..., None]
        samples = (torch.where(mask, ys, -ys + 2 * bd.mean) + pi) % (2 * pi) - pi
        return samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Eq. 2.1 in [1]
        skew_prob = torch.log1p(
            (self.skewness * torch.sin((value - self.base_dist.mean) % (2 * pi))).sum(
                -1
            )
        )
        return self.base_dist.log_prob(value) + skew_prob

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(SineSkewed, _instance)
        base_dist = self.base_dist.expand(batch_shape)
        new.base_dist = base_dist
        new.skewness = self.skewness.expand(batch_shape + (-1,))
        super(SineSkewed, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new
