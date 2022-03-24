# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from pyro.ops.tensor_utils import safe_normalize

from . import constraints
from .torch_distribution import TorchDistribution


class ProjectedNormal(TorchDistribution):
    """
    Projected isotropic normal distribution of arbitrary dimension.

    This distribution over directional data is qualitatively similar to the von
    Mises and von Mises-Fisher distributions, but permits tractable variational
    inference via reparametrized gradients.

    To use this distribution with autoguides, use ``poutine.reparam`` with a
    :class:`~pyro.infer.reparam.projected_normal.ProjectedNormalReparam`
    reparametrizer in the model, e.g.::

        @poutine.reparam(config={"direction": ProjectedNormalReparam()})
        def model():
            direction = pyro.sample("direction",
                                    ProjectedNormal(torch.zeros(3)))
            ...

    or simply wrap in :class:`~pyro.infer.reparam.strategies.MinimalReparam` or
    :class:`~pyro.infer.reparam.strategies.AutoReparam` , e.g.::

        @MinimalReparam()
        def model():
            ...

    .. note:: This implements :meth:`log_prob` only for dimensions {2,3}.

    [1] D. Hernandez-Stumpfhauser, F.J. Breidt, M.J. van der Woerd (2017)
        "The General Projected Normal Distribution of Arbitrary Dimension:
        Modeling and Bayesian Inference"
        https://projecteuclid.org/euclid.ba/1453211962

    :param torch.Tensor concentration: A combined location-and-concentration
        vector. The direction of this vector is the location, and its
        magnitude is the concentration.
    """

    arg_constraints = {"concentration": constraints.real_vector}
    support = constraints.sphere
    has_rsample = True
    _log_prob_impls = {}  # maps dim -> function(concentration, value)

    def __init__(self, concentration, *, validate_args=None):
        assert concentration.dim() >= 1
        self.concentration = concentration
        batch_shape = concentration.shape[:-1]
        event_shape = concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @staticmethod
    def infer_shapes(concentration):
        batch_shape = concentration[:-1]
        event_shape = concentration[-1:]
        return batch_shape, event_shape

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(ProjectedNormal, _instance)
        new.concentration = self.concentration.expand(batch_shape + (-1,))
        super(ProjectedNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self.__dict__.get("_validate_args")
        return new

    @property
    def mean(self):
        """
        Note this is the mean in the sense of a centroid in the submanifold
        that minimizes expected squared geodesic distance.
        """
        return safe_normalize(self.concentration)

    @property
    def mode(self):
        return safe_normalize(self.concentration)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        x = self.concentration.new_empty(shape).normal_()
        x = x + self.concentration
        x = safe_normalize(x)
        return x

    def log_prob(self, value):
        if self._validate_args:
            event_shape = value.shape[-1:]
            if event_shape != self.event_shape:
                raise ValueError(
                    f"Expected event shape {self.event_shape}, "
                    f"but got {event_shape}"
                )
            self._validate_sample(value)
        dim = int(self.concentration.size(-1))
        try:
            impl = self._log_prob_impls[dim]
        except KeyError:
            msg = f"ProjectedNormal.log_prob() is not implemented for dim = {dim}."
            if value.requires_grad:  # For latent variables but not observations.
                msg += " Consider using poutine.reparam with ProjectedNormalReparam."
            raise NotImplementedError(msg)
        return impl(self.concentration, value)

    @classmethod
    def _register_log_prob(cls, dim, fn=None):
        if fn is None:
            return lambda fn: cls._register_log_prob(dim, fn)
        cls._log_prob_impls[dim] = fn
        return fn


def _dot(x, y):
    return (x[..., None, :] @ y[..., None])[..., 0, 0]


@ProjectedNormal._register_log_prob(dim=2)
def _log_prob_2(concentration, value):
    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a univariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t.square()
    r2 = _dot(concentration, concentration) - t2
    perp_part = r2.mul(-0.5) - 0.5 * math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = (t + Sqrt[2/Pi]/E^(t^2/2) + t Erf[t/Sqrt[2]])/2
    para_part = (
        (
            t2.mul(-0.5).exp().mul((2 / math.pi) ** 0.5)
            + t * (1 + (t * 0.5**0.5).erf())
        )
        .mul(0.5)
        .log()
    )

    return para_part + perp_part


@ProjectedNormal._register_log_prob(dim=3)
def _log_prob_3(concentration, value):
    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a bivariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t.square()
    r2 = _dot(concentration, concentration) - t2
    perp_part = r2.mul(-0.5) - math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x^2/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = t/(E^(t^2/2) Sqrt[2 Pi]) + ((1 + t^2) (1 + Erf[t/Sqrt[2]]))/2
    para_part = (
        t * t2.mul(-0.5).exp() / (2 * math.pi) ** 0.5
        + (1 + t2) * (1 + (t * 0.5**0.5).erf()) / 2
    ).log()

    return para_part + perp_part


@ProjectedNormal._register_log_prob(dim=4)
def _log_prob_4(concentration, value):
    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a bivariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t.square()
    r2 = _dot(concentration, concentration) - t2
    perp_part = r2.mul(-0.5) - 1.5 * math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x^3/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = (2 + t^2)/(E^(t^2/2) Sqrt[2 Pi]) + (t (3 + t^2) (1 + Erf[t/Sqrt[2]]))/2
    para_part = (
        (2 + t2) * t2.mul(-0.5).exp() / (2 * math.pi) ** 0.5
        + t * (3 + t2) * (1 + (t * 0.5**0.5).erf()) / 2
    ).log()

    return para_part + perp_part
