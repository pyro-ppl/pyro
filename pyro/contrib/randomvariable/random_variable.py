# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from torch import Tensor
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import (
    Transform,
    AffineTransform,
    AbsTransform,
    PowerTransform,
    ExpTransform,
    TanhTransform,
    SoftmaxTransform,
    SigmoidTransform
)


class RVMagicOps:
    """Mixin class for overloading __magic__ operations on random variables.
    """

    def __add__(self, x: Union[float, Tensor]):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(x, 1)))

    def __radd__(self, x: Union[float, Tensor]):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(x, 1)))

    def __sub__(self, x: Union[float, Tensor]):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(-x, 1)))

    def __rsub__(self, x: Union[float, Tensor]):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(x, -1)))

    def __mul__(self, x: Union[float, Tensor]):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(0, x)))

    def __rmul__(self, x: Union[float, Tensor]):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(0, x)))

    def __truediv__(self, x: Union[float, Tensor]):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(0, 1/x)))

    def __neg__(self):
        return RandomVariable(TransformedDistribution(self.distribution, AffineTransform(0, -1)))

    def __abs__(self):
        return RandomVariable(TransformedDistribution(self.distribution, AbsTransform()))

    def __pow__(self, x):
        return RandomVariable(TransformedDistribution(self.distribution, PowerTransform(x)))


class RVChainOps:
    """Mixin class for performing common unary/binary operations on/between
    random variables/constant tensors using method chaining syntax.
    """

    def add(self, x):
        return self + x

    def sub(self, x):
        return self - x

    def mul(self, x):
        return self * x

    def div(self, x):
        return self / x

    def abs(self):
        return abs(self)

    def pow(self, x):
        return self ** x

    def neg(self):
        return -self

    def exp(self):
        return self.transform(ExpTransform())

    def log(self):
        return self.transform(ExpTransform().inv)

    def sigmoid(self):
        return self.transform(SigmoidTransform())

    def tanh(self):
        return self.transform(TanhTransform())

    def softmax(self):
        return self.transform(SoftmaxTransform())


class RandomVariable(RVMagicOps, RVChainOps):
    """EXPERIMENTAL random variable container class around a distribution

    Representation of a distribution interpreted as a random variable. Rather
    than directly manipulating a probability density by applying pointwise
    transformations to it, this allows for simple arithmetic transformations of
    the random variable the distribution represents. For more flexibility,
    consider using the `transform` method. Note that if you perform a
    non-invertible transform (like `abs(X)` or `X**2`), certain things might
    not work properly.

    Can switch between `RandomVariable` and `Distribution` objects with the
    convenient `Distribution.rv` and `RandomVariable.dist` properties.

    Supports either chaining operations or arithmetic operator overloading.

    Example usage::

        # This should be equivalent to an Exponential distribution.
        RandomVariable(Uniform(0, 1)).log().neg().dist

        # These two distributions Y1, Y2 should be the same
        X = Uniform(0, 1).rv
        Y1 = X.mul(4).pow(0.5).sub(1).abs().neg().dist
        Y2 = (-abs((4*X)**(0.5) - 1)).dist
    """

    def __init__(self, distribution):
        """Wraps a distribution as a RandomVariable

        :param distribution: The `Distribution` object to wrap
        :type distribution: ~pyro.distributions.distribution.Distribution
        """
        self.distribution = distribution

    def transform(self, t: Transform):
        """Performs a transformation on the distribution underlying the RV.

        :param t: The transformation (or sequence of transformations) to be
            applied to the distribution. There are many examples to be found in
            `torch.distributions.transforms` and `pyro.distributions.transforms`,
            or you can subclass directly from `Transform`.
        :type t: ~pyro.distributions.transforms.Transform

        :return: The transformed `RandomVariable`
        :rtype: ~pyro.contrib.randomvariable.random_variable.RandomVariable
        """
        dist = TransformedDistribution(self.distribution, t)
        return RandomVariable(dist)

    @property
    def dist(self):
        """Convenience property for exposing the distribution underlying the
        random variable.

        :return: The `Distribution` object underlying the random variable
        :rtype: ~pyro.distributions.distribution.Distribution
        """
        return self.distribution
