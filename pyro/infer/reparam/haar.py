# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.distributions.transforms.haar import HaarTransform

from .unit_jacobian import UnitJacobianReparam


class HaarReparam(UnitJacobianReparam):
    """
    Haar wavelet reparamterizer, using a
    :class:`~pyro.distributions.transforms.HaarTransform`.

    This is useful for sequential models where coupling along a time-like axis
    (e.g. a banded precision matrix) introduces long-range correlation. This
    reparameterizes to a frequency-domain represetation where posterior
    covariance should be closer to diagonal, thereby improving the accuracy of
    diagonal guides in SVI and improving the effectiveness of a diagonal mass
    matrix in HMC.

    When reparameterizing variables that are approximately continuous along the
    time dimension, set ``smooth=1``. For variables that are approximately
    continuously differentiable along the time axis, set ``smooth=2``.

    This reparameterization works only for latent variables, not likelihoods.

    :param int dim: Dimension along which to transform. Must be negative.
        This is an absolute dim counting from the right.
    :param bool flip: Whether to flip the time axis before applying the
        Haar transform. Defaults to false.
    """
    def __init__(self, dim=-1, flip=False):
        transform = HaarTransform(dim=dim, flip=flip, cache_size=1)
        super().__init__(transform, suffix="haar")
