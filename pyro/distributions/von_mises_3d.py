from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints

from pyro.distributions import TorchDistribution


class VonMises3D(TorchDistribution):
    """
    Spherical von Mises distribution.

    This implementation combines the direction parameter and concentration
    parameter into a single combined parameter that contains both direction and
    magnitude. The ``value`` arg is represented in cartesian coordinates: it
    must be a normalized 3-vector that lies on the 2-sphere.

    See :class:`~pyro.distributions.VonMises` for a 2D polar coordinate cousin
    of this distribution.

    Currently only :meth:`log_prob` is implemented.

    :param torch.Tensor concentration: A combined location-and-concentration
        vector. The direction of this vector is the location, and its
        magnitude is the concentration.
    """
    arg_constraints = {'concentration': constraints.real}
    support = constraints.real  # TODO implement constraints.sphere or similar
    has_rsample = True

    def __init__(self, concentration, validate_args=None):
        if concentration.dim() < 1 or concentration.size(-1) != 3:
            raise ValueError('Expected concentration to have rightmost dim 3, actual shape = {}'.format(
                concentration.shape))
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]

        # Moments
        self._scale = self.concentration.norm(2, -1)
        self._mean = self.concentration / self._scale.unsqueeze(-1)

        # Variance-Covariance Matrix based on:
        # Hillen, Thomas, et al. "Moments of von Mises and Fisher distributions and applications."
        # Mathematical Biosciences & Engineering 14.3 (2017): 673-694.
        cothk = 1 / torch.tanh(self._scale)
        i = torch.eye(3, dtype=self._mean.dtype, device=self._mean.device)
        var_l = (cothk / self._scale - 1 / self._scale ** 2) * i
        mean_outer = self._mean.unsqueeze(-1) * self._mean.unsqueeze(-2)
        var_r = 1 - cothk / self._scale + 2 / self._scale ** 2 - cothk ** 2
        self._variance = var_l + var_r * mean_outer

        # Rotation
        # Based on:
        # Kuba Ober (https://math.stackexchange.com/users/76513/kuba-ober)
        # Calculate Rotation Matrix to align Vector A to Vector B in 3d?
        # URL (version: 2018-09-12): https://math.stackexchange.com/q/897677
        base_tensor = torch.tensor([0., 0., 1.], dtype=self._mean.dtype, device=self._mean.device)
        base_mean_cross = base_tensor.cross(self._mean, dim=-1)
        bmn = base_mean_cross.norm(2, -1)
        base_mean_inner = torch.matmul(base_tensor, self._mean)
        z = torch.zeros_like(base_mean_inner)
        rotg = torch.stack([torch.stack([base_mean_inner, -bmn, z], dim=-1),
                            torch.stack([bmn, base_mean_inner, z], dim=-1),
                            torch.stack([z, z, z+1], dim=-1)], dim=-2)
        vd = (self._mean - base_mean_inner * base_tensor)
        v = vd / vd.norm(2, -1)
        mean_base_cross = self._mean.cross(base_tensor, dim=-1)
        rotfi = torch.stack([base_tensor, v, mean_base_cross], dim=-2)
        self._rotation = torch.eye(3, dtype=self._mean.dtype, device=self._mean.device).expand(batch_shape+(3, 3))
        # If the tensors are on top of each others we do not get a useful rotation matrix.
        # So, we have to change only where they are different
        diffrot = (base_tensor != self._mean).any(-1)
        self._rotation[diffrot] = torch.matmul(torch.matmul(rotfi[diffrot], rotg[diffrot]), rotfi[diffrot].inverse())

        super(VonMises3D, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            if value.dim() < 1 or value.shape[-1] != 3:
                raise ValueError('Expected value to have rightmost dim 3, actual shape = {}'.format(
                    value.shape))
            if not (torch.abs(value.norm(2, -1) - 1) < 1e-6).all():
                raise ValueError('direction vectors are not normalized')
        log_normalizer = self._scale.log() - self._scale.sinh().log() - math.log(4 * math.pi)
        return (self.concentration * value).sum(-1) + log_normalizer

    def expand(self, batch_shape):
        try:
            return super(VonMises3D, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            concentration = self.concentration.expand(torch.Size(batch_shape) + (3,))
            return type(self)(concentration, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        """
        The sampling algorithm for the von Mises-Fisher distribution on the unit sphere (S^2)
        is based on the following paper:
        Jakob, Wenzel. "Numerically stable sampling of the von Mises-Fisher distribution on S^2 (and other tricks)."
        Interactive Geometry Lab, ETH Zuerich, Tech. Rep (2012).
        """
        shape = self._extended_shape(sample_shape)
        vshape = shape[:-1] + (2,)
        wshape = shape[:-1] + (1,)
        x = torch.randn(vshape, dtype=self.concentration.dtype, device=self.concentration.device)
        v = x / x.norm(2, -1).unsqueeze(-1)
        u = torch.rand(wshape, dtype=self.concentration.dtype, device=self.concentration.device)
        kappa = self._scale
        w = 1 + (u + (1 - u) * (-2 * kappa).exp()).log() / kappa
        return torch.matmul(self._rotation, torch.cat([(1 - w ** 2).sqrt() * v, w], dim=-1).unsqueeze(-1)).squeeze()

    @property
    def mean(self):
        """
        The mean direction is on the unit sphere
        """
        return self._mean

    @property
    def variance(self):
        """
        The variance-covariance matrix based on:
        Hillen, Thomas, et al. "Moments of von Mises and Fisher distributions and applications."
        Mathematical Biosciences & Engineering 14.3 (2017): 673-694.
        """
        return self._variance
