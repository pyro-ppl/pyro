from __future__ import absolute_import, division, print_function

import weakref

import torch


def dictionary_is(x, y):
    """
    Helper function that compares the equality of dictionary values.
    """
    return all([k in x and x[k] is y[k] for k in y.keys()])


class Transform(object):
    """
    Abstract class for invertable transformations with computable log det jacobians.
    In contrast to :class:`torch.distributions.transforms.Transform` this class is able
    to represents a transformation that optionally conditions on an observed input.

    Caching is useful for tranforms whose inverses are either expensive or numerically
    unstable.

    Note that care must be taken with memoized values since the autograd graph
    may be reversed. For example while the following works with or without caching:

    y = t(x, obs=z)
    t.log_abs_det_jacobian(x, y, obs=z).backward()`  # x will receive gradients.

    However the following will error when caching due to dependency reversal:

    y = t(x, obs=z)
    a = t.inv(y, obs=z)
    grad(a.sum(), [y])  # error because a is x

    By convention, the observed argument is named for readability, analogously
    to the usage of :meth:`pyro.sample`, although this is not required as any
    keyword argument can be used and will be forward by
    :class:`pyro.distributions.TransformedDistribution`. Derived classes should
    implement one or both of :meth:`_call` or :meth:`_inverse`. Derived classes
    that set `bijective=True` should also implement :meth:`log_abs_det_jacobian`.

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

    def __init__(self, cache_size=0, cond=False):
        self._cache_size = cache_size
        self._inv = None
        self.cond = cond
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y_obs = None, None, None
        else:
            raise ValueError('cache_size must be 0 or 1')
        super(Transform, self).__init__()

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
            inv = _InverseTransform(self)
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

    def __call__(self, x, **kwargs):
        """
        Computes the transform `x | obs => y`.
        """
        if self._cache_size == 0:
            return self._call(x, **kwargs)
        x_old, y_old, obs_old = self._cached_x_y_obs
        if x is x_old and dictionary_is(obs_old, kwargs):
            return y_old
        y = self._call(x, **kwargs)
        self._cached_x_y_obs = x, y, kwargs
        return y

    def _inv_call(self, y, **kwargs):
        """
        Inverts the transform `y | obs => x`.
        """
        if self._cache_size == 0:
            return self._inverse(y, **kwargs)
        x_old, y_old, obs_old = self._cached_x_y_obs
        if y is y_old and dictionary_is(obs_old, kwargs):
            return x_old
        x = self._inverse(y, **kwargs)
        self._cached_x_y_obs = x, y, kwargs
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


class _InverseTransform(torch.distributions.transforms._InverseTransform):
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """

    def __eq__(self, other):
        if not isinstance(other, _InverseTransform):
            return False
        return self._inv == other._inv

    def __call__(self, x, **kwargs):
        return self._inv._inv_call(x, **kwargs)

    def log_abs_det_jacobian(self, x, y, **kwargs):
        return -self._inv.log_abs_det_jacobian(y, x, **kwargs)


class TransformModule(Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.

    """

    def __init__(self, *args, **kwargs):
        super(TransformModule, self).__init__(*args, **kwargs)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()
