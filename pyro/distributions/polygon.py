# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch.distributions.utils import lazy_property

from .torch_distribution import TorchDistribution


class UniformConvexPolygon(TorchDistribution):
    """
    Uniform distribution over a convex polygon.

    .. warning:: This currently implements :meth:`log_prob` for use as a
        likelihod, but does not implement :meth:`sample` for sampling.

    :param torch.Tensor vertices: A ``(num_vertices, 2)`` shaped tensor of
        the vertices of the polygon in clockwise order.
    """

    arg_constraints = {"vertices": constraints.real}  # imprecise

    def __init__(self, vertices: torch.Tensor, *, validate_args: Optional[bool] = None):
        self.vertices = vertices
        batch_shape = vertices.shape[:-2]
        event_shape = torch.Size((2,))
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        if self._validate_args:
            assert vertices.dim() >= 2
            with torch.no_grad():
                x, y = vertices.unbind(-1)
                angles = torch.atan2(y, x)
                if not (angles > angles.roll(1, -1)).all():
                    raise ValueError("Polygon is not convex")

    @constraints.dependent_property
    def support(self):
        return constraints.inside_convex_polygon(self.vertices)

    @lazy_property
    def _log_normalizer(self):
        rays = self.vertices - self.vertices.mean(-2, True)
        a, b = rays.unbind(-1)
        c, d = rays.roll(-2, 1).unbind(-1)
        area = (a * d - b * c).sum(-1)
        return area.log().neg()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.where(self.support.check(value), self._log_normalizer, -math.inf)
