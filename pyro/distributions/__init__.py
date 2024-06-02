# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro.distributions.torch_patch  # noqa F403

# Import * to get the latest upstream distributions.
from pyro.distributions.torch import *  # noqa F403

# Additionally try to import explicitly to help mypy static analysis.
try:
    from pyro.distributions.torch import (
        Bernoulli,
        Beta,
        Binomial,
        Categorical,
        Cauchy,
        Chi2,
        ContinuousBernoulli,
        Dirichlet,
        Exponential,
        ExponentialFamily,
        FisherSnedecor,
        Gamma,
        Geometric,
        Gumbel,
        HalfCauchy,
        HalfNormal,
        Independent,
        Kumaraswamy,
        Laplace,
        LKJCholesky,
        LogisticNormal,
        LogNormal,
        LowRankMultivariateNormal,
        MixtureSameFamily,
        Multinomial,
        MultivariateNormal,
        NegativeBinomial,
        Normal,
        OneHotCategorical,
        OneHotCategoricalStraightThrough,
        Pareto,
        Poisson,
        RelaxedBernoulli,
        RelaxedOneHotCategorical,
        StudentT,
        TransformedDistribution,
        Uniform,
        VonMises,
        Weibull,
        Wishart,
    )
except ImportError:
    pass

from pyro.distributions.affine_beta import AffineBeta
from pyro.distributions.asymmetriclaplace import (
    AsymmetricLaplace,
    SoftAsymmetricLaplace,
)
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
from pyro.distributions.grouped_normal_normal import GroupedNormalNormal
from pyro.distributions.hmm import (
    DiscreteHMM,
    GammaGaussianHMM,
    GaussianHMM,
    GaussianMRF,
    IndependentHMM,
    LinearHMM,
)
from pyro.distributions.improper_uniform import ImproperUniform

if "InverseGamma" not in locals():  # Use PyTorch version if available.
    from pyro.distributions.inverse_gamma import InverseGamma
from pyro.distributions.lkj import LKJ, LKJCorrCholesky
from pyro.distributions.log_normal_negative_binomial import LogNormalNegativeBinomial
from pyro.distributions.logistic import Logistic, SkewLogistic
from pyro.distributions.mixture import MaskedMixture
from pyro.distributions.multivariate_studentt import MultivariateStudentT
from pyro.distributions.nanmasked import NanMaskedMultivariateNormal, NanMaskedNormal
from pyro.distributions.omt_mvn import OMTMultivariateNormal
from pyro.distributions.one_one_matching import OneOneMatching
from pyro.distributions.one_two_matching import OneTwoMatching
from pyro.distributions.ordered_logistic import OrderedLogistic
from pyro.distributions.polya_gamma import TruncatedPolyaGamma
from pyro.distributions.projected_normal import ProjectedNormal
from pyro.distributions.rejector import Rejector
from pyro.distributions.relaxed_straight_through import (
    RelaxedBernoulliStraightThrough,
    RelaxedOneHotCategoricalStraightThrough,
)
from pyro.distributions.sine_bivariate_von_mises import SineBivariateVonMises
from pyro.distributions.sine_skewed import SineSkewed
from pyro.distributions.softlaplace import SoftLaplace
from pyro.distributions.spanning_tree import SpanningTree
from pyro.distributions.stable import Stable, StableWithLogProb
from pyro.distributions.torch import __all__ as torch_dists
from pyro.distributions.torch_distribution import (
    ExpandedDistribution,
    MaskedDistribution,
    TorchDistribution,
)
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
    "AffineBeta",
    "AsymmetricLaplace",
    "Bernoulli",
    "Beta",
    "BetaBinomial",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "CoalescentRateLikelihood",
    "CoalescentTimes",
    "CoalescentTimesWithRate",
    "ComposeTransformModule",
    "ConditionalDistribution",
    "ConditionalTransform",
    "ConditionalTransformModule",
    "ConditionalTransformedDistribution",
    "ContinuousBernoulli",
    "Delta",
    "Dirichlet",
    "DirichletMultinomial",
    "DiscreteHMM",
    "Distribution",
    "Empirical",
    "ExpandedDistribution",
    "Exponential",
    "ExponentialFamily",
    "ExtendedBetaBinomial",
    "ExtendedBinomial",
    "FisherSnedecor",
    "FoldedDistribution",
    "Gamma",
    "GammaGaussianHMM",
    "GammaPoisson",
    "GaussianHMM",
    "GaussianMRF",
    "GaussianScaleMixture",
    "Geometric",
    "GroupedNormalNormal",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "ImproperUniform",
    "Independent",
    "IndependentHMM",
    "InverseGamma",
    "Kumaraswamy",
    "LKJ",
    "LKJCholesky",
    "LKJCorrCholesky",
    "Laplace",
    "LinearHMM",
    "LogNormal",
    "LogNormalNegativeBinomial",
    "Logistic",
    "LogisticNormal",
    "LowRankMultivariateNormal",
    "MaskedDistribution",
    "MaskedMixture",
    "MixtureOfDiagNormals",
    "MixtureOfDiagNormalsSharedCovariance",
    "MixtureSameFamily",
    "Multinomial",
    "MultivariateNormal",
    "MultivariateStudentT",
    "NanMaskedMultivariateNormal",
    "NanMaskedNormal",
    "NegativeBinomial",
    "Normal",
    "OMTMultivariateNormal",
    "OneHotCategorical",
    "OneHotCategoricalStraightThrough",
    "OneOneMatching",
    "OneTwoMatching",
    "OrderedLogistic",
    "Pareto",
    "Poisson",
    "ProjectedNormal",
    "Rejector",
    "RelaxedBernoulli",
    "RelaxedBernoulliStraightThrough",
    "RelaxedOneHotCategorical",
    "RelaxedOneHotCategoricalStraightThrough",
    "SineBivariateVonMises",
    "SineSkewed",
    "SkewLogistic",
    "SoftAsymmetricLaplace",
    "SoftLaplace",
    "SpanningTree",
    "Stable",
    "StableWithLogProb",
    "StudentT",
    "TorchDistribution",
    "TransformModule",
    "TransformedDistribution",
    "TruncatedPolyaGamma",
    "Uniform",
    "Unit",
    "VonMises",
    "VonMises3D",
    "Weibull",
    "Wishart",
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
__all__[:] = sorted(set(__all__))
del torch_dists
