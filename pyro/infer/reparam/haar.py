# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.distributions.transforms.haar import HaarTransform

from .unit_jacobian import UnitJacobianReparam


class HaarReparam(UnitJacobianReparam):
    """
    Haar wavelet reparameterizer, using a
    :class:`~pyro.distributions.transforms.HaarTransform`.

    This is useful for sequential models where coupling along a time-like axis
    (e.g. a banded precision matrix) introduces long-range correlation. This
    reparameterizes to a frequency-domain representation where posterior
    covariance should be closer to diagonal, thereby improving the accuracy of
    diagonal guides in SVI and improving the effectiveness of a diagonal mass
    matrix in HMC.

    This reparameterization works only for latent variables, not likelihoods.

    :param int dim: Dimension along which to transform. Must be negative.
        This is an absolute dim counting from the right.
    :param bool flip: Whether to flip the time axis before applying the
        Haar transform. Defaults to false.
    """
    def __init__(self, dim=-1, flip=False, *,
                 experimental_allow_batch=False):
        transform = HaarTransform(dim=dim, flip=flip, cache_size=1)
        super().__init__(transform, suffix="haar",
                         experimental_allow_batch=experimental_allow_batch)
