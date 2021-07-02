# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions.transforms import *  # noqa F403

# isort: split

from torch.distributions import biject_to, transform_to
from torch.distributions.transforms import (
    ComposeTransform,
    ExpTransform,
    LowerCholeskyTransform,
)
from torch.distributions.transforms import __all__ as torch_transforms

from .. import constraints
from ..torch_transform import ComposeTransformModule
from .affine_autoregressive import (
    AffineAutoregressive,
    ConditionalAffineAutoregressive,
    affine_autoregressive,
    conditional_affine_autoregressive,
)
from .affine_coupling import (
    AffineCoupling,
    ConditionalAffineCoupling,
    affine_coupling,
    conditional_affine_coupling,
)
from .basic import ELUTransform, LeakyReLUTransform, elu, leaky_relu
from .batchnorm import BatchNorm, batchnorm
from .block_autoregressive import BlockAutoregressive, block_autoregressive
from .cholesky import (
    CholeskyTransform,
    CorrLCholeskyTransform,
    CorrMatrixCholeskyTransform,
)
from .discrete_cosine import DiscreteCosineTransform
from .generalized_channel_permute import (
    ConditionalGeneralizedChannelPermute,
    GeneralizedChannelPermute,
    conditional_generalized_channel_permute,
    generalized_channel_permute,
)
from .haar import HaarTransform
from .householder import (
    ConditionalHouseholder,
    Householder,
    conditional_householder,
    householder,
)
from .lower_cholesky_affine import LowerCholeskyAffine
from .matrix_exponential import (
    ConditionalMatrixExponential,
    MatrixExponential,
    conditional_matrix_exponential,
    matrix_exponential,
)
from .neural_autoregressive import (
    ConditionalNeuralAutoregressive,
    NeuralAutoregressive,
    conditional_neural_autoregressive,
    neural_autoregressive,
)
from .normalize import Normalize
from .ordered import OrderedTransform
from .permute import Permute, permute
from .planar import ConditionalPlanar, Planar, conditional_planar, planar
from .polynomial import Polynomial, polynomial
from .radial import ConditionalRadial, Radial, conditional_radial, radial
from .softplus import SoftplusLowerCholeskyTransform, SoftplusTransform
from .spline import ConditionalSpline, Spline, conditional_spline, spline
from .spline_autoregressive import (
    ConditionalSplineAutoregressive,
    SplineAutoregressive,
    conditional_spline_autoregressive,
    spline_autoregressive,
)
from .spline_coupling import SplineCoupling, spline_coupling
from .sylvester import Sylvester, sylvester

########################################
# register transforms


@transform_to.register(constraints.sphere)
def _transform_to_sphere(constraint):
    return Normalize()


@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return CorrLCholeskyTransform()


@biject_to.register(constraints.corr_matrix)
@transform_to.register(constraints.corr_matrix)
def _transform_to_corr_matrix(constraint):
    return ComposeTransform(
        [CorrLCholeskyTransform(), CorrMatrixCholeskyTransform().inv]
    )


@biject_to.register(constraints.ordered_vector)
@transform_to.register(constraints.ordered_vector)
def _transform_to_ordered_vector(constraint):
    return OrderedTransform()


@biject_to.register(constraints.positive_ordered_vector)
@transform_to.register(constraints.positive_ordered_vector)
def _transform_to_positive_ordered_vector(constraint):
    return ComposeTransform([OrderedTransform(), ExpTransform()])


# TODO: register biject_to when LowerCholeskyTransform is bijective
@transform_to.register(constraints.positive_definite)
def _transform_to_positive_definite(constraint):
    return ComposeTransform([LowerCholeskyTransform(), CholeskyTransform().inv])


@biject_to.register(constraints.softplus_positive)
@transform_to.register(constraints.softplus_positive)
def _transform_to_softplus_positive(constraint):
    return SoftplusTransform()


@transform_to.register(constraints.softplus_lower_cholesky)
def _transform_to_softplus_lower_cholesky(constraint):
    return SoftplusLowerCholeskyTransform()


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
    "iterated",
    "AffineAutoregressive",
    "AffineCoupling",
    "BatchNorm",
    "BlockAutoregressive",
    "CholeskyTransform",
    "ComposeTransformModule",
    "ConditionalAffineAutoregressive",
    "ConditionalAffineCoupling",
    "ConditionalGeneralizedChannelPermute",
    "ConditionalHouseholder",
    "ConditionalMatrixExponential",
    "ConditionalNeuralAutoregressive",
    "ConditionalPlanar",
    "ConditionalRadial",
    "ConditionalSpline",
    "ConditionalSplineAutoregressive",
    "CorrLCholeskyTransform",
    "CorrMatrixCholeskyTransform",
    "DiscreteCosineTransform",
    "ELUTransform",
    "GeneralizedChannelPermute",
    "HaarTransform",
    "Householder",
    "LeakyReLUTransform",
    "LowerCholeskyAffine",
    "MatrixExponential",
    "NeuralAutoregressive",
    "Normalize",
    "OrderedTransform",
    "Permute",
    "Planar",
    "Polynomial",
    "Radial",
    "SoftplusLowerCholeskyTransform",
    "SoftplusTransform",
    "Spline",
    "SplineAutoregressive",
    "SplineCoupling",
    "Sylvester",
    "affine_autoregressive",
    "affine_coupling",
    "batchnorm",
    "block_autoregressive",
    "conditional_affine_autoregressive",
    "conditional_affine_coupling",
    "conditional_generalized_channel_permute",
    "conditional_householder",
    "conditional_matrix_exponential",
    "conditional_neural_autoregressive",
    "conditional_planar",
    "conditional_radial",
    "conditional_spline",
    "conditional_spline_autoregressive",
    "elu",
    "generalized_channel_permute",
    "householder",
    "leaky_relu",
    "matrix_exponential",
    "neural_autoregressive",
    "permute",
    "planar",
    "polynomial",
    "radial",
    "spline",
    "spline_autoregressive",
    "spline_coupling",
    "sylvester",
]

__all__.extend(torch_transforms)
del torch_transforms
