# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.distributions.transforms.discrete_cosine import DiscreteCosineTransform

from .unit_jacobian import UnitJacobianReparam


class DiscreteCosineReparam(UnitJacobianReparam):
    """
    Discrete Cosine reparameterizer, using a
    :class:`~pyro.distributions.transforms.DiscreteCosineTransform` .

    This is useful for sequential models where coupling along a time-like axis
    (e.g. a banded precision matrix) introduces long-range correlation. This
    reparameterizes to a frequency-domain representation where posterior
    covariance should be closer to diagonal, thereby improving the accuracy of
    diagonal guides in SVI and improving the effectiveness of a diagonal mass
    matrix in HMC.

    When reparameterizing variables that are approximately continuous along the
    time dimension, set ``smooth=1``. For variables that are approximately
    continuously differentiable along the time axis, set ``smooth=2``.

    This reparameterization works only for latent variables, not likelihoods.

    :param int dim: Dimension along which to transform. Must be negative.
        This is an absolute dim counting from the right.
    :param float smooth: Smoothing parameter. When 0, this transforms white
        noise to white noise; when 1 this transforms Brownian noise to to white
        noise; when -1 this transforms violet noise to white noise; etc. Any
        real number is allowed. https://en.wikipedia.org/wiki/Colors_of_noise.
    """
    def __init__(self, dim=-1, smooth=0., *,
                 experimental_allow_batch=False):
        transform = DiscreteCosineTransform(dim=dim, smooth=smooth, cache_size=1)
        super().__init__(transform, suffix="dct",
                         experimental_allow_batch=experimental_allow_batch)
