from __future__ import absolute_import

import weakref

import torch
from torch.distributions.utils import _sum_rightmost

from pyro.distributions.transforms.batch_norm import BatchNormTransform
from pyro.distributions.transforms.householder import HouseholderFlow
from pyro.distributions.transforms.iaf import InverseAutoregressiveFlow, InverseAutoregressiveFlowStable
from pyro.distributions.transforms.naf import DeepELUFlow, DeepLeakyReLUFlow, DeepSigmoidalFlow
from pyro.distributions.transforms.permute import PermuteTransform
from pyro.distributions.transforms.polynomial import PolynomialFlow
from pyro.distributions.transforms.planar import PlanarFlow
from pyro.distributions.transforms.radial import RadialFlow
from pyro.distributions.transforms.sylvester import SylvesterFlow


__all__ = [
    'BatchNormTransform',
    'DeepELUFlow',
    'DeepLeakyReLUFlow',
    'DeepSigmoidalFlow',
    'HouseholderFlow',
    'InverseAutoregressiveFlow',
    'InverseAutoregressiveFlowStable',
    'PermuteTransform',
    'PolynomialFlow',
    'PlanarFlow',
    'RadialFlow',
    'SylvesterFlow',
]


class ConditionalTransform(object):
    """
    Abstract class for invertable conditional transformations with computable
    log det jacobians. In contrast to :class:`torch.distributions.transforms.Transform`
    this class represents a transformation that conditions on an observed input.

    Caching is useful for tranforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works with or without caching::

        y = t(x, obs=z)
        t.log_abs_det_jacobian(x, y, obs=z).backward()  # x will receive gradients.

    However the following will error when caching due to dependency reversal::

        y = t(x, obs=z)
        a = t.inv(y, obs=z)
        grad(a.sum(), [y])  # error because a is x

    By convention, the observed argument is named for readability, analogously
    to the usage of :meth:`pyro.sample`, although this is not required.

    Derived classes should implement one or both of :meth:`_call` or
    :meth:`_inverse`. Derived classes that set `bijective=True` should also
    implement :meth:`log_abs_det_jacobian`.

    Args:
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.

    Attributes:
        domain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid inputs to this transform.
        codomain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid outputs to this transform
            which are inputs to the inverse transform.
        bijective (bool): Whether this transform is bijective. A transform
            ``t`` is bijective iff ``t.inv(t(x)) == x`` and
            ``t(t.inv(y)) == y`` for every ``x`` in the domain and ``y`` in
            the codomain. Transforms that are not bijective should at least
            maintain the weaker pseudoinverse properties
            ``t(t.inv(t(x)) == t(x)`` and ``t.inv(t(t.inv(y))) == t.inv(y)``.
        sign (int or Tensor): For bijective univariate transforms, this
            should be +1 or -1 depending on whether transform is monotone
            increasing or decreasing.
        event_dim (int): Number of dimensions that are correlated together in
            the transform ``event_shape``. This should be 0 for pointwise
            transforms, 1 for transforms that act jointly on vectors, 2 for
            transforms that act jointly on matrices, etc.
    """
    bijective = False
    event_dim = 0

    def __init__(self, cache_size=0):
        self._cache_size = cache_size
        self._inv = None
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y_obs = None, None, None
        else:
            raise ValueError('cache_size must be 0 or 1')
        super(ConditionalTransform, self).__init__()

    @property
    def inv(self):
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseConditionalTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    @property
    def sign(self):
        """
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        """
        raise NotImplementedError

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        # Necessary for Python2
        return not self.__eq__(other)

    def __call__(self, x, obs):
        """
        Computes the transform `x | obs => y`.
        """
        if self._cache_size == 0:
            return self._call(x, obs)
        x_old, y_old, obs_old = self._cached_x_y_obs
        if x is x_old and obs is obs_old:
            return y_old
        y = self._call(x, obs)
        self._cached_x_y_obs = x, y, obs
        return y

    def _inv_call(self, y, obs):
        """
        Inverts the transform `y | obs => x`.
        """
        if self._cache_size == 0:
            return self._inverse(y, obs)
        x_old, y_old, obs_old = self._cached_x_y_obs
        if y is y_old and obs is obs_old:
            return x_old
        x = self._inverse(y, obs)
        self._cached_x_y_obs = x, y, obs
        return x

    def _call(self, x, obs):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def _inverse(self, y, obs):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y, obs):
        """
        Computes the log det jacobian `log |dy/dx|` given input, output, and conditioned variable.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class _InverseConditionalTransform(torch.distributions.transforms._InverseTransform):
    """
    Inverts a single :class:`ConditionalTransform`.
    This class is private; please instead use the ``ConditionalTransform.inv`` property.
    """

    def __eq__(self, other):
        if not isinstance(other, _InverseConditionalTransform):
            return False
        return self._inv == other._inv

    def __call__(self, x, obs):
        return self._inv._inv_call(x, obs)

    def log_abs_det_jacobian(self, x, y, obs):
        return -self._inv.log_abs_det_jacobian(y, x, obs)


class ComposeConditionalTransform(torch.distributions.ComposeTransform, ConditionalTransform):
    """
    Composes multiple transforms or conditional transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform` and/or :class:`ConditionalTransform`): A list of transforms to compose.
    """

    def __call__(self, x, obs):
        for part in self.parts:
            if isinstance(part, torch.distributions.Transform):
                x = part(x)
            else:
                x = part(x, obs)
        return x

    def log_abs_det_jacobian(self, x, y, obs):
        if not self.parts:
            return torch.zeros_like(x)
        result = 0
        for part in self.parts[:-1]:
            if isinstance(part, torch.distributions.Transform):
                y_tmp = part(x)
                result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y_tmp),
                                                 self.event_dim - part.event_dim)
            else:
                y_tmp = part(x, obs)
                result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y_tmp, obs),
                                                 self.event_dim - part.event_dim)

            x = y_tmp
        part = self.parts[-1]
        if isinstance(part, torch.distributions.Transform):
            result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y),
                                             self.event_dim - part.event_dim)
        else:
            result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y, obs),
                                             self.event_dim - part.event_dim)
        return result

    def __repr__(self):
        fmt_string = self.__class__.__name__ + '(\n    '
        fmt_string += ',\n    '.join([p.__repr__() for p in self.parts])
        fmt_string += '\n)'
        return fmt_string
