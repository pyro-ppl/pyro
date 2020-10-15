# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import biject_to, transform_to
from torch.distributions.transforms import *  # noqa F403
from torch.distributions.transforms import __all__ as torch_transforms

from pyro.distributions.constraints import (
                                            IndependentConstraint,
                                            corr_cholesky_constraint,
                                            ordered_vector)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.transforms.affine_autoregressive import (AffineAutoregressive, ConditionalAffineAutoregressive,
                                                                 affine_autoregressive,
                                                                 conditional_affine_autoregressive)
from pyro.distributions.transforms.affine_coupling import (AffineCoupling, ConditionalAffineCoupling, affine_coupling,
                                                           conditional_affine_coupling)
from pyro.distributions.transforms.basic import ELUTransform, LeakyReLUTransform, elu, leaky_relu
from pyro.distributions.transforms.batchnorm import BatchNorm, batchnorm
from pyro.distributions.transforms.block_autoregressive import BlockAutoregressive, block_autoregressive
from pyro.distributions.transforms.cholesky import CorrLCholeskyTransform
from pyro.distributions.transforms.discrete_cosine import DiscreteCosineTransform
from pyro.distributions.transforms.generalized_channel_permute import (ConditionalGeneralizedChannelPermute,
                                                                       GeneralizedChannelPermute,
                                                                       conditional_generalized_channel_permute,
                                                                       generalized_channel_permute)
from pyro.distributions.transforms.haar import HaarTransform
from pyro.distributions.transforms.householder import (ConditionalHouseholder, Householder, conditional_householder,
                                                       householder)
from pyro.distributions.transforms.lower_cholesky_affine import LowerCholeskyAffine
from pyro.distributions.transforms.matrix_exponential import (ConditionalMatrixExponential, MatrixExponential,
                                                              conditional_matrix_exponential, matrix_exponential)
from pyro.distributions.transforms.neural_autoregressive import (ConditionalNeuralAutoregressive, NeuralAutoregressive,
                                                                 conditional_neural_autoregressive,
                                                                 neural_autoregressive)
from pyro.distributions.transforms.ordered import OrderedTransform
from pyro.distributions.transforms.permute import Permute, permute
from pyro.distributions.transforms.planar import ConditionalPlanar, Planar, conditional_planar, planar
from pyro.distributions.transforms.polynomial import Polynomial, polynomial
from pyro.distributions.transforms.radial import ConditionalRadial, Radial, conditional_radial, radial
from pyro.distributions.transforms.spline import ConditionalSpline, Spline, conditional_spline, spline
from pyro.distributions.transforms.spline_autoregressive import (ConditionalSplineAutoregressive, SplineAutoregressive,
                                                                 conditional_spline_autoregressive,
                                                                 spline_autoregressive)
from pyro.distributions.transforms.spline_coupling import SplineCoupling, spline_coupling
from pyro.distributions.transforms.sylvester import Sylvester, sylvester

########################################
# register transforms

biject_to.register(IndependentConstraint, lambda c: biject_to(c.base_constraint))
transform_to.register(IndependentConstraint, lambda c: transform_to(c.base_constraint))


@biject_to.register(corr_cholesky_constraint)
@transform_to.register(corr_cholesky_constraint)
def _transform_to_corr_cholesky(constraint):
    return CorrLCholeskyTransform()


@biject_to.register(ordered_vector)
@transform_to.register(ordered_vector)
def _transform_to_ordered_vector(constraint):
    return OrderedTransform()


def iterated(repeats, base_fn, *args, **kwargs):
    """
    Helper function to compose a sequence of bijective transforms with potentially
    learnable parameters using :class:`~pyro.distributions.ComposeTransformModule`.

    :param repeats: number of repeated transforms.
    :param base_fn: function to construct the bijective transform.
    :param args: arguments taken by `base_fn`.
    :param kwargs: keyword arguments taken by `base_fn`.
    :return: instance of :class:`~pyro.distributions.TransformModule`.
    """
    assert isinstance(repeats, int) and repeats >= 1
    return ComposeTransformModule([base_fn(*args, **kwargs) for _ in range(repeats)])


__all__ = [
    'iterated',
    'AffineAutoregressive',
    'AffineCoupling',
    'BatchNorm',
    'BlockAutoregressive',
    'ComposeTransformModule',
    'ConditionalAffineAutoregressive',
    'ConditionalAffineCoupling',
    'ConditionalGeneralizedChannelPermute',
    'ConditionalHouseholder',
    'ConditionalMatrixExponential',
    'ConditionalNeuralAutoregressive',
    'ConditionalPlanar',
    'ConditionalRadial',
    'ConditionalSpline',
    'ConditionalSplineAutoregressive',
    'CorrLCholeskyTransform',
    'DiscreteCosineTransform',
    'ELUTransform',
    'GeneralizedChannelPermute',
    'HaarTransform',
    'Householder',
    'LeakyReLUTransform',
    'LowerCholeskyAffine',
    'MatrixExponential',
    'NeuralAutoregressive',
    'OrderedTransform',
    'Permute',
    'Planar',
    'Polynomial',
    'Radial',
    'Spline',
    'SplineAutoregressive',
    'SplineCoupling',
    'Sylvester',
    'affine_autoregressive',
    'affine_coupling',
    'batchnorm',
    'block_autoregressive',
    'conditional_affine_autoregressive',
    'conditional_affine_coupling',
    'conditional_generalized_channel_permute',
    'conditional_householder',
    'conditional_matrix_exponential',
    'conditional_neural_autoregressive',
    'conditional_planar',
    'conditional_radial',
    'conditional_spline',
    'conditional_spline_autoregressive',
    'elu',
    'generalized_channel_permute',
    'householder',
    'leaky_relu',
    'matrix_exponential',
    'neural_autoregressive',
    'permute',
    'planar',
    'polynomial',
    'radial',
    'spline',
    'spline_autoregressive',
    'spline_coupling',
    'sylvester',
]

__all__.extend(torch_transforms)
del torch_transforms
