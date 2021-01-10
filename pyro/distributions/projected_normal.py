# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from . import constraints
from .torch_distribution import TorchDistribution
from .util import broadcast_shape


def safe_project(x):
    """
    Safely project a vector onto the sphere. This avoid the singularity at zero
    by mapping to the vector ``[1, 0, 0, ..., 0]``.

    :param Tensor x: A vector
    :returns: A normalized version ``x / ||x||_2``.
    :rtype: Tensor
    """
    x = x / torch.linalg.norm(x, dim=-1).clamp(min=torch.finfo(x.dtype).tiny)
    x.data[..., 0][x.data.eq(0).all(dim=-1)] = 1  # Avoid the singularity.
    return x


class ProjectedNormal(TorchDistribution):
    """
    Projected isotropic normal distribution of arbitrary dimension.

    This distribution over directional data is qualitatively similar to the von
    Mises and von Mises-Fisher distributions, but permits tractable variational
    inference via reparametrized gradients.

    To use this distribution with autoguides, use ``poutine.reparam`` with a
    :class:`~pyro.infer.reparam.projectednormal.ProjectedNormalReparam`
    reparametrizer in the model, e.g.::

        @poutine.reparam(config={"direction": ProjectedNormalReparam()})
        def model():
            direction = pyro.sample("direction", ProjectedNormal(torch.zeros(3)))
            ...

    .. note:: This implements :meth:`log_prob` only for dimensions {2,3}.

    [1] D. Hernandez-Stumpfhauser, F.J. Breidt, M.J. van der Woerd (2017)
        "The General Projected Normal Distribution of Arbitrary Dimension:
        Modeling and Bayesian Inference"
        https://projecteuclid.org/euclid.ba/1453211962
    """
    arg_constraints = {"concentration": constraints.real_vector}
    support = constraints.sphere
    _log_prob_impls = {}

    def __init__(self, concentration, *, validate_args=None):
        self.concentration = concentration
        batch_shape = concentration.shape[:-1]
        event_shape = concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ProjectedNormal, _instance)
        batch_shape = torch.Size(broadcast_shape(self.batch_shape, batch_shape))
        new.concentration = self.concentration.expand(batch_shape + (-1,))
        super(ProjectedNormal, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get('_validate_args')
        return new

    @property
    def mean(self):
        # Note this is the mean in the sense of a centroid
        # that minimizes expected squared geodesic distance.
        return safe_project(self.concentration)

    @property
    def mode(self):
        return safe_project(self.concentration)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        x = self.concentration.new_empty(shape).normal_()
        x = x + self.concentration
        x = safe_project(x)
        return x

    def log_prob(self, value):
        if self._validate_args:
            event_shape = value.shape[:-1]
            if event_shape != self.event_shape:
                raise ValueError(f"Expected event shape {self.event_shape}, "
                                 f"but got {event_shape}")
            self._validate_sample(value)
        dim = int(self.concentration.size(-1))
        try:
            impl = self._log_prob_impls[dim]
        except KeyError:
            raise NotImplementedError(
                f"ProjectedNormal.log_prob() is not implemented for dim = {dim}. "
                "Consider using poutine.reparam with ProjectedNormalReparam.")
        return impl(self.concentration, value)

    @classmethod
    def _register_log_prob(cls, dim, fn=None):
        if fn is None:
            return lambda fn: cls._register_log_prob(dim, fn)
        cls._log_prob_impls[dim] = fn
        return fn


def _dot(x, y):
    return (x[..., None, :] @ y[..., None]).squeeze([-2, -1])


@ProjectedNormal._register_log_prob(dim=2)
def _log_prob_2(concentration, value):
    c = concentration
    x = value
    cx = _dot(c, x)
    cx2 = cx.square()
    # This corresponds to the mathematica definite integral
    # Integrate[x/(E^((x-c)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    para_part = (cx2.mul(-0.5).exp().mul((2 / math.pi) ** 0.5)
                 + cx + cx * (cx * 0.5 ** 0.5).erf()).mul(0.5).log()

    c_perp_x = _dot(c, c) - cx2
    perp_part = c_perp_x.mul(-0.5).exp() - 0.5 * math.log(2 * math.pi)

    return para_part + perp_part


@ProjectedNormal._register_log_prob(dim=3)
def _log_prob_3(concentration, value):
    c = concentration
    x = value
    cx = _dot(c, x)
    cx2 = cx.square()
    # This corresponds to the mathematica definite integral
    # Integrate[x^2/(E^((x-c)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    para_part = (0.5 * (cx2 + 1) * (1 + (cx * 0.5 ** 0.5).erf())
                 + cx2.mul(-0.5).exp() * c / (2 * math.pi) ** 0.5).log()

    c_perp_x = _dot(c, c) - cx2
    perp_part = c_perp_x.mul(-0.5).exp() - math.log(2 * math.pi)

    return para_part + perp_part
