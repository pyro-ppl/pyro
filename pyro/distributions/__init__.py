# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro.distributions.torch_patch  # noqa F403
from pyro.distributions.avf_mvn import AVFMultivariateNormal
from pyro.distributions.coalescent import (
    CoalescentRateLikelihood,
    CoalescentTimes,
    CoalescentTimesWithRate,
)
from pyro.distributions.conditional import (
    ConditionalDistribution,
    ConditionalTransform,
    ConditionalTransformedDistribution,
    ConditionalTransformModule,
)
from pyro.distributions.conjugate import (
    BetaBinomial,
    DirichletMultinomial,
    GammaPoisson,
)
from pyro.distributions.delta import Delta
from pyro.distributions.diag_normal_mixture import MixtureOfDiagNormals
from pyro.distributions.diag_normal_mixture_shared_cov import (
    MixtureOfDiagNormalsSharedCovariance,
)
from pyro.distributions.distribution import Distribution
from pyro.distributions.empirical import Empirical
from pyro.distributions.extended import ExtendedBetaBinomial, ExtendedBinomial
from pyro.distributions.folded import FoldedDistribution
from pyro.distributions.gaussian_scale_mixture import GaussianScaleMixture
from pyro.distributions.hmm import (
    DiscreteHMM,
    GammaGaussianHMM,
    GaussianHMM,
    GaussianMRF,
    IndependentHMM,
    LinearHMM,
)
from pyro.distributions.improper_uniform import ImproperUniform
from pyro.distributions.inverse_gamma import InverseGamma
from pyro.distributions.lkj import LKJCorrCholesky
from pyro.distributions.mixture import MaskedMixture
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from pyro.distributions.omt_mvn import OMTMultivariateNormal
from pyro.distributions.ordered_logistic import OrderedLogistic
from pyro.distributions.polya_gamma import TruncatedPolyaGamma
from pyro.distributions.rejector import Rejector
from pyro.distributions.relaxed_straight_through import (
    RelaxedBernoulliStraightThrough,
    RelaxedOneHotCategoricalStraightThrough,
)
from pyro.distributions.spanning_tree import SpanningTree
from pyro.distributions.stable import Stable
from pyro.distributions.torch import *  # noqa F403
from pyro.distributions.torch import __all__ as torch_dists
from pyro.distributions.torch_distribution import MaskedDistribution, TorchDistribution
from pyro.distributions.torch_transform import ComposeTransformModule, TransformModule
from pyro.distributions.unit import Unit
from pyro.distributions.util import (
    enable_validation,
    is_validation_enabled,
    validation_enabled,
)
from pyro.distributions.von_mises_3d import VonMises3D
from pyro.distributions.zero_inflated import (
    ZeroInflatedDistribution,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)

from . import constraints, kl, transforms

__all__ = [
    "AVFMultivariateNormal",
    "BetaBinomial",
    "CoalescentRateLikelihood",
    "CoalescentTimes",
    "CoalescentTimesWithRate",
    "ComposeTransformModule",
    "ConditionalDistribution",
    "ConditionalTransform",
    "ConditionalTransformModule",
    "ConditionalTransformedDistribution",
    "Delta",
    "DirichletMultinomial",
    "DiscreteHMM",
    "Distribution",
    "Empirical",
    "ExtendedBetaBinomial",
    "ExtendedBinomial",
    "FoldedDistribution",
    "GammaGaussianHMM",
    "GammaPoisson",
    "GaussianHMM",
    "GaussianMRF",
    "GaussianScaleMixture",
    "ImproperUniform",
    "IndependentHMM",
    "InverseGamma",
    "LinearHMM",
    "LKJCorrCholesky",
    "MaskedDistribution",
    "MaskedMixture",
    "MixtureOfDiagNormals",
    "MixtureOfDiagNormalsSharedCovariance",
    "MultivariateStudentT",
    "OMTMultivariateNormal",
    "OrderedLogistic",
    "Rejector",
    "RelaxedBernoulliStraightThrough",
    "RelaxedOneHotCategoricalStraightThrough",
    "SpanningTree",
    "Stable",
    "TorchDistribution",
    "TransformModule",
    "TruncatedPolyaGamma",
    "Unit",
    "VonMises3D",
    "ZeroInflatedDistribution",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
    "constraints",
    "enable_validation",
    "is_validation_enabled",
    "kl",
    "transforms",
    "validation_enabled",
]

# Import all torch distributions from `pyro.distributions.torch_distribution`
__all__.extend(torch_dists)
del torch_dists
