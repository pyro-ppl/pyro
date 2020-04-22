# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import biject_to
from torch.distributions.transforms import ComposeTransform

import pyro
import pyro.distributions as dist
from pyro.distributions.transforms.discrete_cosine import DiscreteCosineTransform

from .reparam import Reparam


class DiscreteCosineReparam(Reparam):
    """
    Discrete Cosine reparamterizer, using a
    :class:`~pyro.distributions.transforms.DiscreteCosineTransform` .

    This is useful for sequential models where coupling along a time-like axis
    (e.g. a banded precision matrix) introduces long-range correlation. This
    reparameterizes to a frequency-domain represetation where posterior
    covariance should be closer to diagonal, thereby improving the accuracy of
    diagonal guides in SVI and improving the effectiveness of a diagonal mass
    matrix in HMC.

    This reparameterization works only for latent variables, not likelihoods.

    :param int dim: Dimension along which to transform. Must be negative.
        This is an absolute dim counting from the right.
    """
    def __init__(self, dim=-1, smooth=0.):
        assert isinstance(dim, int) and dim < 0
        self.dim = dim
        self.smooth = float(smooth)

    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        assert fn.event_dim >= -self.dim, ("Cannot transform along batch dimension; "
                                           "try converting a batch dimension to an event dimension")

        # Draw noise from the base distribution.
        # TODO Use biject_to(fn.support).inv.with_cache(1) once the following merges:
        # https://github.com/probtorch/pytorch/pull/153
        dct = DiscreteCosineTransform(dim=self.dim, smooth=self.smooth, cache_size=1)
        transform = ComposeTransform([biject_to(fn.support).inv, dct])
        x_dct = pyro.sample("{}_dct".format(name),
                            dist.TransformedDistribution(fn, transform))

        # Differentiably transform.
        x = transform.inv(x_dct)  # should be free due to transform cache

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(x, event_dim=fn.event_dim)
        return new_fn, x
