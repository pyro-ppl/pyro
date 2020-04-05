# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import biject_to, transform_to
from torch.distributions.transforms import *  # noqa F403
from torch.distributions.transforms import __all__ as torch_transforms

from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.transforms.affine_autoregressive import AffineAutoregressive, affine_autoregressive
from pyro.distributions.transforms.affine_coupling import (AffineCoupling, ConditionalAffineCoupling,
                                                           affine_coupling, conditional_affine_coupling)
from pyro.distributions.transforms.batchnorm import BatchNorm, batchnorm
from pyro.distributions.transforms.block_autoregressive import BlockAutoregressive, block_autoregressive
from pyro.distributions.transforms.discrete_cosine import DiscreteCosineTransform
from pyro.distributions.transforms.generalized_channel_permute import (GeneralizedChannelPermute,
                                                                       generalized_channel_permute)
from pyro.distributions.transforms.householder import Householder, householder
from pyro.distributions.transforms.lower_cholesky_affine import LowerCholeskyAffine
from pyro.distributions.transforms.neural_autoregressive import (ELUTransform, LeakyReLUTransform,
                                                                 TanhTransform, NeuralAutoregressive,
                                                                 neural_autoregressive,
                                                                 elu, leaky_relu, tanh)
from pyro.distributions.transforms.permute import Permute, permute
from pyro.distributions.transforms.polynomial import Polynomial, polynomial
from pyro.distributions.transforms.planar import Planar, ConditionalPlanar, planar, conditional_planar
from pyro.distributions.transforms.radial import Radial, ConditionalRadial, radial, conditional_radial
from pyro.distributions.transforms.spline import Spline, spline
from pyro.distributions.transforms.sylvester import Sylvester, sylvester
from pyro.distributions.constraints import IndependentConstraint, corr_cholesky_constraint
from pyro.distributions.transforms.cholesky import CorrLCholeskyTransform

########################################
# register transforms

biject_to.register(IndependentConstraint, lambda c: biject_to(c.base_constraint))
transform_to.register(IndependentConstraint, lambda c: transform_to(c.base_constraint))


@biject_to.register(corr_cholesky_constraint)
@transform_to.register(corr_cholesky_constraint)
def _transform_to_corr_cholesky(constraint):
    return CorrLCholeskyTransform()


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
    'ConditionalAffineCoupling',
    'ConditionalPlanar',
    'ConditionalRadial',
    'CorrLCholeskyTransform',
    'DiscreteCosineTransform',
    'ELUTransform',
    'GeneralizedChannelPermute',
    'Householder',
    'LeakyReLUTransform',
    'LowerCholeskyAffine',
    'NeuralAutoregressive',
    'Permute',
    'Planar',
    'Polynomial',
    'Radial',
    'Spline',
    'Sylvester',
    'TanhTransform',
    'affine_autoregressive',
    'affine_coupling',
    'batchnorm',
    'block_autoregressive',
    'conditional_affine_coupling',
    'conditional_planar',
    'conditional_radial',
    'elu',
    'generalized_channel_permute',
    'householder',
    'leaky_relu',
    'neural_autoregressive',
    'permute',
    'planar',
    'polynomial',
    'radial',
    'spline',
    'sylvester',
    'tanh',
]

__all__.extend(torch_transforms)
del torch_transforms
