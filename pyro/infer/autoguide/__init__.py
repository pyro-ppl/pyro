# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pyro.infer.autoguide.effect import (
    AutoMessenger,
    AutoNormalMessenger,
    AutoHierarchicalNormal,
    AutoRegressiveMessenger,
)
from pyro.infer.autoguide.gaussian import AutoGaussian
from pyro.infer.autoguide.guides import (
    AutoCallable,
    AutoContinuous,
    AutoDelta,
    AutoDiagonalNormal,
    AutoDiscreteParallel,
    AutoGuide,
    AutoGuideList,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoNormalizingFlow,
)
from pyro.infer.autoguide.initialization import (
    init_to_feasible,
    init_to_generated,
    init_to_mean,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)
from pyro.infer.autoguide.structured import AutoStructured
from pyro.infer.autoguide.utils import mean_field_entropy

__all__ = [
    "AutoCallable",
    "AutoContinuous",
    "AutoDelta",
    "AutoDiagonalNormal",
    "AutoDiscreteParallel",
    "AutoGaussian",
    "AutoGuide",
    "AutoGuideList",
    "AutoIAFNormal",
    "AutoLaplaceApproximation",
    "AutoLowRankMultivariateNormal",
    "AutoMessenger",
    "AutoMultivariateNormal",
    "AutoNormal",
    "AutoNormalMessenger",
    "AutoHierarchicalNormal",
    "AutoNormalizingFlow",
    "AutoRegressiveMessenger",
    "AutoStructured",
    "init_to_feasible",
    "init_to_generated",
    "init_to_mean",
    "init_to_median",
    "init_to_sample",
    "init_to_uniform",
    "init_to_value",
    "mean_field_entropy",
]
