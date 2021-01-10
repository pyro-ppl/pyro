# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

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

    [1] D. Hernandez-Stumpfhauser, F.J. Breidt, M.J. van der Woerd (2017)
        "The General Projected Normal Distribution of Arbitrary Dimension:
        Modeling and Bayesian Inference"
        https://projecteuclid.org/euclid.ba/1453211962
    """
    arg_constraints = {"concentration": constraints.real_vector}
    support = constraints.sphere

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

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        x = self.concentration.new_empty(shape).normal_()
        x = x + self.concentration
        x = safe_project(x)
        return x

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        raise NotImplementedError("Use poutine.reparam with ProjectedNormalReparam")

    @property
    def mode(self):
        return safe_project(self.concentration)
