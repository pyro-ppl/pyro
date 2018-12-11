from __future__ import absolute_import, division, print_function

import pyro.distributions.torch_patch  # noqa F403
from pyro.distributions.avf_mvn import AVFMultivariateNormal
from pyro.distributions.delta import Delta
from pyro.distributions.diag_normal_mixture_shared_cov import MixtureOfDiagNormalsSharedCovariance
from pyro.distributions.diag_normal_mixture import MixtureOfDiagNormals
from pyro.distributions.distribution import Distribution
from pyro.distributions.empirical import Empirical
from pyro.distributions.gaussian_scale_mixture import GaussianScaleMixture
from pyro.distributions.iaf import InverseAutoregressiveFlow, InverseAutoregressiveFlowStable
from pyro.distributions.mixture import MaskedMixture
from pyro.distributions.omt_mvn import OMTMultivariateNormal
from pyro.distributions.permute import PermuteTransform
from pyro.distributions.planar import PlanarFlow
from pyro.distributions.rejector import Rejector
from pyro.distributions.relaxed_straight_through import (RelaxedBernoulliStraightThrough,
                                                         RelaxedOneHotCategoricalStraightThrough)
from pyro.distributions.torch import *  # noqa F403
from pyro.distributions.torch import __all__ as torch_dists
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import enable_validation, is_validation_enabled, validation_enabled
from pyro.distributions.von_mises import VonMises
from pyro.distributions.von_mises_3d import VonMises3D
from pyro.distributions.zero_inflated_poisson import ZeroInflatedPoisson

__all__ = [
    "enable_validation",
    "is_validation_enabled",
    "validation_enabled",
    "AVFMultivariateNormal",
    "Delta",
    "Distribution",
    "Empirical",
    "GaussianScaleMixture",
    "InverseAutoregressiveFlow",
    "InverseAutoregressiveFlowStable",
    "MaskedMixture",
    "MixtureOfDiagNormalsSharedCovariance",
    "MixtureOfDiagNormals",
    "OMTMultivariateNormal",
    "PermuteTransform",
    "PlanarFlow",
    "Rejector",
    "RelaxedBernoulliStraightThrough",
    "RelaxedOneHotCategoricalStraightThrough",
    "TorchDistribution",
    "TransformModule",
    "VonMises",
    "VonMises3D",
    "ZeroInflatedPoisson"
]

# Import all torch distributions from `pyro.distributions.torch_distribution`
__all__.extend(torch_dists)
del torch_dists
