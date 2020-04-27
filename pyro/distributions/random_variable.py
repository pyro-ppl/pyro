# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Callable

from torch import Tensor
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import (
    Transform,
    AffineTransform,
    AbsTransform,
    PowerTransform
)
from pyro.distributions import Distribution


class RVArithmeticMixin:
    """Mixin class for overloading arithmetic operations on random variables
    """

    def __add__(self, x: Union[float, Tensor]):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(x, 1))
        )

    def __radd__(self, x: Union[float, Tensor]):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(x, 1))
        )

    def __sub__(self, x: Union[float, Tensor]):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(-x, 1))
        )

    def __rsub__(self, x: Union[float, Tensor]):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(x, -1))
        )

    def __mul__(self, x: Union[float, Tensor]):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(0, x))
        )

    def __rmul__(self, x: Union[float, Tensor]):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(0, x))
        )

    def __truediv__(self, x: Union[float, Tensor]):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(0, 1/x))
        )

    def __neg__(self):
        return RandomVariable(
            TransformedDistribution(self.distribution, AffineTransform(0, -1))
        )

    def __abs__(self):
        return RandomVariable(
            TransformedDistribution(self.distribution, AbsTransform())
        )

    def __pow__(self, x):
        return RandomVariable(
            TransformedDistribution(self.distribution, PowerTransform(x))
        )


class RandomVariable(RVArithmeticMixin):
    """Random variable container class around a distribution

    Representation of a distribution interpreted as a random variable. Rather
    than directly manipulating a probability density by applying pointwise
    transformations to it, this allows for simple arithmetic transformations of
    the random variable the distribution represents. For more flexibility,
    consider using the `transform` method. Note that if you
    perform a non-invertible transform (like `abs(X)` or `X**2`), certain
    things might not work properly.
    """

    def __init__(self, distribution: Distribution):
        self.distribution = distribution

    def __getattr__(self, name):
        return self.distribution.__getattribute__(name)

    def __call__(self, *args, **kwargs):
        return self.distribution(*args, **kwargs)

    def tranform(self, t: Transform):
        """Performs a transformation on the distribution underlying the RV.

        :param t: The transformation (or sequence of transformations) to be 
            applied to the distribution. There are many examples to be found in
            `torch.distributions.transforms` and `pyro.distributions.transforms`,
            or you can subclass directly from `Transform`.
        :type t: `Transform`
        """
        dist = TransformedDistribution(self.distribution, t)
        return RandomVariable(dist)
