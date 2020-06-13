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
    :param experimental_event_dim: EXPERIMENTAL Optional ``event_dim``,
        overriding the default of ``event_dim = -dim``. This results in a
        proper transform only if ``event_dim >= -dim``; however an improper
        transform (that mixes elements across batches) can still be used in
        some applications, such as reparameterization without subsampling.
    """
    def __init__(self, dim=-1, flip=False, *, experimental_event_dim=None):
        transform = HaarTransform(dim=dim, flip=flip, cache_size=1,
                                  experimental_event_dim=experimental_event_dim)
        super().__init__(transform, suffix="haar")
