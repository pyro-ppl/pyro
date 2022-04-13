# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


class InsideConvexPolygonTransform(Transform):
    """
    Transforms an arbitrary 2D point to the interior of a convex polygon.
    """

    bijective = True
    domain = constraints.real_vector

    def __init__(self, vertices: torch.Tensor, cache_size=0):
        super().__init__(cache_size=cache_size)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return type(self)(self.vertices, cache_size=cache_size)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.inside_convex_polygon(self.vertices)

    def _call(self, x):
        raise NotImplementedError("TODO")

    def _inverse(self, y):
        raise NotImplementedError("TODO")

    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError("TODO")
