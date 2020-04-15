# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
import scipy.stats as sp

import pyro.distributions as dist
from pyro.distributions.testing.naive_dirichlet import NaiveBeta, NaiveDirichlet
from pyro.distributions.testing.rejection_exponential import RejectionExponential
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedBeta, ShapeAugmentedDirichlet, ShapeAugmentedGamma
from tests.distributions.dist_fixture import Fixture


class FoldedNormal(dist.FoldedDistribution):
    def __init__(self, loc, scale):
        super().__init__(dist.Normal(loc, scale))


continuous_dists = [
    Fixture(pyro_dist=dist.Uniform,
            scipy_dist=sp.uniform,
            examples=[
                {'low': [2.], 'high': [2.5],
                 'test_data': [2.2]},
                {'low': [2., 4.], 'high': [3., 5.],
                 'test_data': [[[2.5, 4.5]], [[2.5, 4.5]], [[2.5, 4.5]]]},
                {'low': [[2.], [-3.], [0.]],
                 'high': [[2.5], [0.], [1.]],
                 'test_data': [[2.2], [-2], [0.7]]},
            ],
            scipy_arg_fn=lambda low, high: ((), {"loc": np.array(low),
                                                 "scale": np.array(high) - np.array(low)})),
    Fixture(pyro_dist=dist.Exponential,
            scipy_dist=sp.expon,
            examples=[
                {'rate': [2.4],
                 'test_data': [5.5]},
                {'rate': [2.4, 5.5],
                 'test_data': [[[5.5, 3.2]], [[5.5, 3.2]], [[5.5, 3.2]]]},
                {'rate': [[2.4, 5.5]],
                 'test_data': [[[5.5, 3.2]], [[5.5, 3.2]], [[5.5, 3.2]]]},
                {'rate': [[2.4], [5.5]],
                 'test_data': [[5.5], [3.2]]},
            ],
            scipy_arg_fn=lambda rate: ((), {"scale": 1.0 / np.array(rate)})),
    Fixture(pyro_dist=RejectionExponential,
            scipy_dist=sp.expon,
            examples=[
                {'rate': [2.4], 'factor': [0.5],
                 'test_data': [5.5]},
                {'rate': [2.4, 5.5], 'factor': [0.5],
                 'test_data': [[[5.5, 3.2]], [[5.5, 3.2]], [[5.5, 3.2]]]},
                {'rate': [[2.4, 5.5]], 'factor': [0.5],
                 'test_data': [[[5.5, 3.2]], [[5.5, 3.2]], [[5.5, 3.2]]]},
                {'rate': [[2.4], [5.5]], 'factor': [0.5],
                 'test_data': [[5.5], [3.2]]},
            ],
            scipy_arg_fn=lambda rate, factor: ((), {"scale": 1.0 / np.array(rate)})),
    Fixture(pyro_dist=dist.Gamma,
            scipy_dist=sp.gamma,
            examples=[
                {'concentration': [2.4], 'rate': [3.2],
                 'test_data': [5.5]},
                {'concentration': [[2.4, 2.4], [3.2, 3.2]], 'rate': [[2.4, 2.4], [3.2, 3.2]],
                 'test_data': [[[5.5, 4.4], [5.5, 4.4]]]},
                {'concentration': [[2.4], [2.4]], 'rate': [[3.2], [3.2]], 'test_data': [[5.5], [4.4]]}
            ],
            scipy_arg_fn=lambda concentration, rate: ((np.array(concentration),),
                                                      {"scale": 1.0 / np.array(rate)})),
    Fixture(pyro_dist=ShapeAugmentedGamma,
            scipy_dist=sp.gamma,
            examples=[
                {'concentration': [2.4], 'rate': [3.2],
                 'test_data': [5.5]},
                {'concentration': [[2.4, 2.4], [3.2, 3.2]], 'rate': [[2.4, 2.4], [3.2, 3.2]],
                 'test_data': [[[5.5, 4.4], [5.5, 4.4]]]},
                {'concentration': [[2.4], [2.4]], 'rate': [[3.2], [3.2]], 'test_data': [[5.5], [4.4]]}
            ],
            scipy_arg_fn=lambda concentration, rate: ((np.array(concentration),),
                                                      {"scale": 1.0 / np.array(rate)})),
    Fixture(pyro_dist=dist.Beta,
            scipy_dist=sp.beta,
            examples=[
                {'concentration1': [2.4], 'concentration0': [3.6],
                 'test_data': [0.4]},
                {'concentration1': [[2.4, 2.4], [3.6, 3.6]], 'concentration0': [[2.5, 2.5], [2.5, 2.5]],
                 'test_data': [[[0.5, 0.4], [0.5, 0.4]]]},
                {'concentration1': [[2.4], [3.7]], 'concentration0': [[3.6], [2.5]],
                 'test_data': [[0.4], [0.6]]}
            ],
            scipy_arg_fn=lambda concentration1, concentration0:
            ((np.array(concentration1), np.array(concentration0)), {})),

    Fixture(pyro_dist=NaiveBeta,
            scipy_dist=sp.beta,
            examples=[
                {'concentration1': [2.4], 'concentration0': [3.6],
                 'test_data': [0.4]},
                {'concentration1': [[2.4, 2.4], [3.6, 3.6]], 'concentration0': [[2.5, 2.5], [2.5, 2.5]],
                 'test_data': [[[0.5, 0.4], [0.5, 0.4]]]},
                {'concentration1': [[2.4], [3.7]], 'concentration0': [[3.6], [2.5]],
                 'test_data': [[0.4], [0.6]]}
            ],
            scipy_arg_fn=lambda concentration1, concentration0:
            ((np.array(concentration1), np.array(concentration0)), {})),
    Fixture(pyro_dist=ShapeAugmentedBeta,
            scipy_dist=sp.beta,
            examples=[
                {'concentration1': [2.4], 'concentration0': [3.6],
                 'test_data': [0.4]},
                {'concentration1': [[2.4, 2.4], [3.6, 3.6]], 'concentration0': [[2.5, 2.5], [2.5, 2.5]],
                 'test_data': [[[0.5, 0.4], [0.5, 0.4]]]},
                {'concentration1': [[2.4], [3.7]], 'concentration0': [[3.6], [2.5]],
                 'test_data': [[0.4], [0.6]]}
            ],
            scipy_arg_fn=lambda concentration1, concentration0:
            ((np.array(concentration1), np.array(concentration0)), {})),
    Fixture(pyro_dist=dist.LogNormal,
            scipy_dist=sp.lognorm,
            examples=[
                {'loc': [1.4], 'scale': [0.4],
                 'test_data': [5.5]},
                {'loc': [1.4], 'scale': [0.4],
                 'test_data': [[5.5]]},
                {'loc': [[1.4, 0.4], [1.4, 0.4]], 'scale': [[2.6, 0.5], [2.6, 0.5]],
                 'test_data': [[5.5, 6.4], [5.5, 6.4]]},
                {'loc': [[1.4], [0.4]], 'scale': [[2.6], [0.5]],
                 'test_data': [[5.5], [6.4]]}
            ],
            scipy_arg_fn=lambda loc, scale: ((np.array(scale),), {"scale": np.exp(np.array(loc))})),
    Fixture(pyro_dist=dist.Normal,
            scipy_dist=sp.norm,
            examples=[
                {'loc': [2.0], 'scale': [4.0],
                 'test_data': [2.0]},
                {'loc': [[2.0]], 'scale': [[4.0]],
                 'test_data': [[2.0]]},
                {'loc': [[[2.0]]], 'scale': [[[4.0]]],
                 'test_data': [[[2.0]]]},
                {'loc': [2.0, 50.0], 'scale': [4.0, 100.0],
                 'test_data': [[2.0, 50.0], [2.0, 50.0]]},
            ],
            scipy_arg_fn=lambda loc, scale: ((), {"loc": np.array(loc), "scale": np.array(scale)}),
            prec=0.07,
            min_samples=50000),
    Fixture(pyro_dist=dist.MultivariateNormal,
            scipy_dist=sp.multivariate_normal,
            examples=[
                {'loc': [2.0, 1.0], 'covariance_matrix': [[1.0, 0.5], [0.5, 1.0]],
                 'test_data': [[2.0, 1.0], [9.0, 3.4]]},
            ],
            # This hack seems to be the best option right now, as 'scale' is not handled well by get_scipy_batch_logpdf
            scipy_arg_fn=lambda loc, covariance_matrix=None:
                ((), {"mean": np.array(loc), "cov": np.array([[1.0, 0.5], [0.5, 1.0]])}),
            prec=0.01,
            min_samples=500000),
    Fixture(pyro_dist=dist.LowRankMultivariateNormal,
            scipy_dist=sp.multivariate_normal,
            examples=[
                {'loc': [2.0, 1.0], 'cov_diag': [0.5, 0.5], 'cov_factor': [[1.0], [0.5]],
                 'test_data': [[2.0, 1.0], [9.0, 3.4]]},
            ],
            scipy_arg_fn=lambda loc, cov_diag=None, cov_factor=None:
                ((), {"mean": np.array(loc), "cov": np.array([[1.5, 0.5], [0.5, 0.75]])}),
            prec=0.01,
            min_samples=500000),
    Fixture(pyro_dist=FoldedNormal,
            examples=[
                {'loc': [2.0], 'scale': [4.0],
                 'test_data': [2.0]},
                {'loc': [[2.0]], 'scale': [[4.0]],
                 'test_data': [[2.0]]},
                {'loc': [[[2.0]]], 'scale': [[[4.0]]],
                 'test_data': [[[2.0]]]},
                {'loc': [2.0, 50.0], 'scale': [4.0, 100.0],
                 'test_data': [[2.0, 50.0], [2.0, 50.0]]},
            ]),
    Fixture(pyro_dist=dist.Dirichlet,
            scipy_dist=sp.dirichlet,
            examples=[
                {'concentration': [2.4, 3, 6],
                 'test_data': [0.2, 0.45, 0.35]},
                {'concentration': [2.4, 3, 6],
                 'test_data': [[0.2, 0.45, 0.35], [0.2, 0.45, 0.35]]},
                {'concentration': [[2.4, 3, 6], [3.2, 1.2, 0.4]],
                 'test_data': [[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]]}
            ],
            scipy_arg_fn=lambda concentration: ((concentration,), {})),
    Fixture(pyro_dist=NaiveDirichlet,
            scipy_dist=sp.dirichlet,
            examples=[
                {'concentration': [2.4, 3, 6],
                 'test_data': [0.2, 0.45, 0.35]},
                {'concentration': [2.4, 3, 6],
                 'test_data': [[0.2, 0.45, 0.35], [0.2, 0.45, 0.35]]},
                {'concentration': [[2.4, 3, 6], [3.2, 1.2, 0.4]],
                 'test_data': [[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]]}
            ],
            scipy_arg_fn=lambda concentration: ((concentration,), {})),
    Fixture(pyro_dist=ShapeAugmentedDirichlet,
            scipy_dist=sp.dirichlet,
            examples=[
                {'concentration': [2.4, 3, 6],
                 'test_data': [0.2, 0.45, 0.35]},
                {'concentration': [2.4, 3, 6],
                 'test_data': [[0.2, 0.45, 0.35], [0.2, 0.45, 0.35]]},
                {'concentration': [[2.4, 3, 6], [3.2, 1.2, 0.4]],
                 'test_data': [[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]]}
            ],
            scipy_arg_fn=lambda concentration: ((concentration,), {})),
    Fixture(pyro_dist=dist.Cauchy,
            scipy_dist=sp.cauchy,
            examples=[
                {'loc': [0.5], 'scale': [1.2],
                 'test_data': [1.0]},
                {'loc': [0.5, 0.5], 'scale': [1.2, 1.2],
                 'test_data': [[1.0, 1.0], [1.0, 1.0]]},
                {'loc': [[0.5], [0.3]], 'scale': [[1.2], [1.0]],
                 'test_data': [[0.4], [0.35]]}
            ],
            scipy_arg_fn=lambda loc, scale: ((), {"loc": np.array(loc), "scale": np.array(scale)})),
    Fixture(pyro_dist=dist.HalfCauchy,
            scipy_dist=sp.halfcauchy,
            examples=[
                {'scale': [1.2],
                 'test_data': [1.0]},
                {'scale': [1.2, 1.2],
                 'test_data': [[1.0, -1.0], [1.0, -1.0]]},
                {'scale': [[1.2], [1.0]],
                 'test_data': [[0.54], [0.35]]}
            ],
            scipy_arg_fn=lambda scale: ((), {"scale": np.array(scale)})),
    Fixture(pyro_dist=dist.VonMises,
            scipy_dist=sp.vonmises,
            examples=[
                {'loc': [0.5], 'concentration': [1.2],
                 'test_data': [1.0]},
                {'loc': [0.5, 3.0], 'concentration': [2.0, 0.5],
                 'test_data': [[1.0, 2.0], [1.0, 2.0]]},
                {'loc': [[0.5], [0.3]], 'concentration': [[2.0], [0.5]],
                 'test_data': [[1.0], [2.0]]}
            ],
            scipy_arg_fn=lambda loc, concentration: ((), {"loc": np.array(loc), "kappa": np.array(concentration)})),
    Fixture(pyro_dist=dist.LKJCorrCholesky,
            examples=[
                {'d': 3, 'eta': [1.], 'test_data':
                    [[[1.0000,  0.0000,  0.0000], [-0.8221,  0.5693,  0.0000], [0.7655,  0.1756,  0.6190]],
                     [[1.0000,  0.0000,  0.0000], [-0.5345,  0.8451,  0.0000], [-0.5459, -0.3847,  0.7444]],
                     [[1.0000,  0.0000,  0.0000], [-0.3758,  0.9267,  0.0000], [-0.2409,  0.4044,  0.8823]],
                     [[1.0000,  0.0000,  0.0000], [-0.8800,  0.4750,  0.0000], [-0.9493,  0.1546,  0.2737]],
                     [[1.0000,  0.0000,  0.0000], [0.2284,  0.9736,  0.0000], [-0.1283,  0.0451,  0.9907]]]},
                ]),
    Fixture(pyro_dist=dist.Stable,
            examples=[
                {'stability': [1.5], 'skew': 0.1, 'test_data': [-10.]},
                {'stability': [1.5], 'skew': 0.1, 'scale': 2.0, 'loc': -2.0, 'test_data': [10.]},
                ]),
    Fixture(pyro_dist=dist.MultivariateStudentT,
            examples=[
                {'df': 1.5, 'loc': [0.2, 0.3], 'scale_tril': [[0.8, 0.0], [1.3, 0.4]],
                 'test_data': [-3., 2]},
                ]),
]

discrete_dists = [
    Fixture(pyro_dist=dist.Multinomial,
            scipy_dist=sp.multinomial,
            examples=[
                {'probs': [0.1, 0.6, 0.3],
                 'test_data': [0., 1., 0.]},
                {'probs': [0.1, 0.6, 0.3], 'total_count': 8,
                 'test_data': [2., 4., 2.]},
                {'probs': [0.1, 0.6, 0.3], 'total_count': 8,
                 'test_data': [[2., 4., 2.], [2., 4., 2.]]},
                {'probs': [[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]], 'total_count': 8,
                 'test_data': [[2., 4., 2.], [1., 4., 3.]]}
            ],
            scipy_arg_fn=lambda probs, total_count=[1]: ((total_count[0], np.array(probs)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.Bernoulli,
            scipy_dist=sp.bernoulli,
            examples=[
                {'probs': [0.25],
                 'test_data': [1.]},
                {'probs': [0.25, 0.25],
                 'test_data': [[[0., 1.]], [[1., 0.]], [[0., 0.]]]},
                {'logits': [math.log(p / (1 - p)) for p in (0.25, 0.25)],
                 'test_data': [[[0., 1.]], [[1., 0.]], [[0., 0.]]]},
                # for now, avoid tests on infinite logits
                # {'logits': [-float('inf'), 0],
                #  'test_data': [[0, 1], [0, 1], [0, 1]]},
                {'logits': [[math.log(p / (1 - p)) for p in (0.25, 0.25)],
                            [math.log(p / (1 - p)) for p in (0.3, 0.3)]],
                 'test_data': [[1., 1.], [0., 0.]]},
                {'probs': [[0.25, 0.25], [0.3, 0.3]],
                 'test_data': [[1., 1.], [0., 0.]]}
            ],
            # for now, avoid tests on infinite logits
            # test_data_indices=[0, 1, 2, 3],
            batch_data_indices=[-1, -2],
            scipy_arg_fn=lambda **kwargs: ((), {'p': kwargs['probs']}),
            prec=0.01,
            min_samples=10000,
            is_discrete=True,
            expected_support_non_vec=[[0.], [1.]],
            expected_support=[[[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]]]),
    Fixture(pyro_dist=dist.BetaBinomial,
            examples=[
                {'concentration1': [2.], 'concentration0': [5.], 'total_count': 8,
                 'test_data': [4.]},
                {'concentration1': [2.], 'concentration0': [5.], 'total_count': 8,
                 'test_data': [[2.], [4.]]},
                {'concentration1': [[2.], [2.]], 'concentration0': [[5.], [5.]], 'total_count': 8,
                 'test_data': [[4.], [3.]]},
                {'concentration1': [2., 2.], 'concentration0': [5., 5.], 'total_count': [0., 0.],
                 'test_data': [[0., 0.], [0., 0.]]},
                {'concentration1': [2., 2.], 'concentration0': [5., 5.], 'total_count': [[8., 7.], [5., 9.]],
                 'test_data': [[6., 3.], [2., 8.]]},
            ],
            batch_data_indices=[-1, -2],
            prec=0.01,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.Binomial,
            scipy_dist=sp.binom,
            examples=[
                {'probs': [0.6], 'total_count': 8,
                 'test_data': [4.]},
                {'probs': [0.3], 'total_count': 8,
                 'test_data': [[2.], [4.]]},
                {'probs': [[0.2], [0.4]], 'total_count': 8,
                 'test_data': [[4.], [3.]]},
                {'probs': [0.2, 0.4], 'total_count': [0., 0.],
                 'test_data': [[0., 0.], [0., 0.]]},
                {'probs': [0.2, 0.4], 'total_count': [[8., 7.], [5., 9.]],
                 'test_data': [[6., 3.], [2., 8.]]},
            ],
            scipy_arg_fn=lambda probs, total_count: ((total_count, probs), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.ExtendedBetaBinomial,
            examples=[
                {'concentration1': [2.], 'concentration0': [5.], 'total_count': 8,
                 'test_data': [4.]},
                {'concentration1': [2.], 'concentration0': [5.], 'total_count': 8,
                 'test_data': [[2.], [4.]]},
                {'concentration1': [[2.], [2.]], 'concentration0': [[5.], [5.]], 'total_count': 8,
                 'test_data': [[4.], [3.]]},
                {'concentration1': [2., 2.], 'concentration0': [5., 5.], 'total_count': [0., 0.],
                 'test_data': [[0., 0.], [0., 0.]]},
                {'concentration1': [2., 2.], 'concentration0': [5., 5.], 'total_count': [[8., 7.], [5., 9.]],
                 'test_data': [[6., 3.], [2., 8.]]},
            ],
            batch_data_indices=[-1, -2],
            prec=0.01,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.ExtendedBinomial,
            scipy_dist=sp.binom,
            examples=[
                {'probs': [0.6], 'total_count': 8,
                 'test_data': [4.]},
                {'probs': [0.3], 'total_count': 8,
                 'test_data': [[2.], [4.]]},
                {'probs': [[0.2], [0.4]], 'total_count': 8,
                 'test_data': [[4.], [3.]]},
                {'probs': [0.2, 0.4], 'total_count': [0., 0.],
                 'test_data': [[0., 0.], [0., 0.]]},
                {'probs': [0.2, 0.4], 'total_count': [[8., 7.], [5., 9.]],
                 'test_data': [[6., 3.], [2., 8.]]},
            ],
            scipy_arg_fn=lambda probs, total_count: ((total_count, probs), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.Categorical,
            scipy_dist=sp.multinomial,
            examples=[
                {'probs': [0.1, 0.6, 0.3],
                 'test_data': [2]},
                {'logits': list(map(math.log, [0.1, 0.6, 0.3])),
                 'test_data': [2]},
                {'logits': [list(map(math.log, [0.1, 0.6, 0.3])),
                            list(map(math.log, [0.2, 0.4, 0.4]))],
                 'test_data': [2, 0]},
                {'probs': [[0.1, 0.6, 0.3],
                           [0.2, 0.4, 0.4]],
                 'test_data': [2, 0]}
            ],
            test_data_indices=[0, 1, 2],
            batch_data_indices=[-1, -2],
            scipy_arg_fn=None,
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.DirichletMultinomial,
            examples=[
                {'concentration': [0.1, 0.6, 0.3],
                 'test_data': [0., 1., 0.]},
                {'concentration': [0.5, 1.0, 2.0], 'total_count': 8,
                 'test_data': [0., 2., 6.]},
                {'concentration': [[0.5, 1.0, 2.0], [3., 3., 0.1]], 'total_count': 8,
                 'test_data': [[0., 2., 6.], [5., 2., 1.]]},
            ],
            prec=0.08,
            is_discrete=True),
    Fixture(pyro_dist=dist.GammaPoisson,
            examples=[
                {'concentration': [1.], 'rate': [2.],
                 'test_data': [0.]},
                {'concentration': [1.], 'rate': [2.],
                 'test_data': [1.]},
                {'concentration': [1.], 'rate': [2.],
                 'test_data': [4.]},
                {'concentration': [1., 1., 1.], 'rate': [2., 2., 3.],
                 'test_data': [[0., 1., 4.], [0., 1., 4.]]},
                {'concentration': [[1.0], [1.0], [1.0]], 'rate': [[2.0], [2.0], [3.0]],
                 'test_data': [[0.], [1.], [4.]]}
            ],
            prec=0.08,
            is_discrete=True),
    Fixture(pyro_dist=dist.OneHotCategorical,
            scipy_dist=sp.multinomial,
            examples=[
                {'probs': [0.1, 0.6, 0.3],
                 'test_data': [0., 0., 1.]},
                {'logits': list(map(math.log, [0.1, 0.6, 0.3])),
                 'test_data': [0., 0., 1.]},
                {'logits': [list(map(math.log, [0.1, 0.6, 0.3])),
                            list(map(math.log, [0.2, 0.4, 0.4]))],
                 'test_data': [[0., 0., 1.], [1., 0., 0.]]},
                {'probs': [[0.1, 0.6, 0.3],
                           [0.2, 0.4, 0.4]],
                 'test_data': [[0., 0., 1.], [1., 0., 0.]]}
            ],
            test_data_indices=[0, 1, 2],
            batch_data_indices=[-1, -2],
            scipy_arg_fn=lambda probs: ((1, np.array(probs)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.Poisson,
            scipy_dist=sp.poisson,
            examples=[
                {'rate': [2.0],
                 'test_data': [0.]},
                {'rate': [3.0],
                 'test_data': [1.]},
                {'rate': [6.0],
                 'test_data': [4.]},
                {'rate': [2.0, 3.0, 6.0],
                 'test_data': [[0., 1., 4.], [0., 1., 4.]]},
                {'rate': [[2.0], [3.0], [6.0]],
                 'test_data': [[0.], [1.], [4.]]}
            ],
            scipy_arg_fn=lambda rate: ((np.array(rate),), {}),
            prec=0.08,
            is_discrete=True),
    Fixture(pyro_dist=dist.Geometric,
            scipy_dist=sp.geom,
            examples=[
                {'logits': [2.0],
                 'test_data': [0.]},
                {'logits': [3.0],
                 'test_data': [1.]},
                {'logits': [-6.0],
                 'test_data': [4.]},
                {'logits': [2.0, 3.0, -6.0],
                 'test_data': [[0., 1., 4.], [0., 1., 4.]]},
                {'logits': [[2.0], [3.0], [-6.0]],
                 'test_data': [[0.], [1.], [4.]]}
            ],
            scipy_arg_fn=lambda probs: ((np.array(probs), -1), {}),
            prec=0.08,
            is_discrete=True),
]


@pytest.fixture(name='dist',
                params=continuous_dists + discrete_dists,
                ids=lambda x: x.get_test_distribution_name())
def all_distributions(request):
    return request.param


@pytest.fixture(name='discrete_dist',
                params=discrete_dists,
                ids=lambda x: x.get_test_distribution_name())
def discrete_distributions(request):
    return request.param


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/distributions"):
            if "stage" not in item.keywords:
                item.add_marker(pytest.mark.stage("unit"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
