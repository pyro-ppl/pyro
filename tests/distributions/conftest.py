from __future__ import absolute_import, division, print_function

import math

import numpy as np
import pytest
import scipy.stats as sp

import pyro.distributions as dist
from pyro.distributions.testing.naive_dirichlet import NaiveBeta, NaiveDirichlet
from pyro.distributions.testing.rejection_exponential import RejectionExponential
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedBeta, ShapeAugmentedDirichlet, ShapeAugmentedGamma
from tests.distributions.dist_fixture import Fixture

continuous_dists = [
    Fixture(pyro_dist=dist.Uniform,
            scipy_dist=sp.uniform,
            examples=[
                {'a': [2], 'b': [2.5],
                 'test_data': [2.2]},
                {'a': [2, 4], 'b': [3, 5],
                 'test_data': [[[2.5, 4.5]], [[2.5, 4.5]], [[2.5, 4.5]]]},
                {'a': [[2], [-3], [0]],
                 'b': [[2.5], [0], [1]],
                 'test_data': [[2.2], [-2], [0.7]]},
            ],
            scipy_arg_fn=lambda a, b: ((), {"loc": np.array(a),
                                            "scale": np.array(b) - np.array(a)})),
    Fixture(pyro_dist=dist.Exponential,
            scipy_dist=sp.expon,
            examples=[
                {'lam': [2.4],
                 'test_data': [5.5]},
                {'lam': [2.4, 5.5],
                 'test_data': [[[5.5, 3.2]], [[5.5, 3.2]], [[5.5, 3.2]]]},
                {'lam': [[2.4, 5.5]],
                 'test_data': [[[5.5, 3.2]], [[5.5, 3.2]], [[5.5, 3.2]]]},
                {'lam': [[2.4], [5.5]],
                 'test_data': [[5.5], [3.2]]},
            ],
            scipy_arg_fn=lambda lam: ((), {"scale": 1.0 / np.array(lam)})),
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
                {'alpha': [2.4], 'beta': [3.2],
                 'test_data': [5.5]},
                {'alpha': [[2.4, 2.4], [3.2, 3.2]], 'beta': [[2.4, 2.4], [3.2, 3.2]],
                 'test_data': [[[5.5, 4.4], [5.5, 4.4]]]},
                {'alpha': [[2.4], [2.4]], 'beta': [[3.2], [3.2]], 'test_data': [[5.5], [4.4]]}
            ],
            scipy_arg_fn=lambda alpha, beta: ((np.array(alpha),),
                                              {"scale": 1.0 / np.array(beta)})),
    Fixture(pyro_dist=ShapeAugmentedGamma,
            scipy_dist=sp.gamma,
            examples=[
                {'alpha': [2.4], 'beta': [3.2],
                 'test_data': [5.5]},
                {'alpha': [[2.4, 2.4], [3.2, 3.2]], 'beta': [[2.4, 2.4], [3.2, 3.2]],
                 'test_data': [[[5.5, 4.4], [5.5, 4.4]]]},
                {'alpha': [[2.4], [2.4]], 'beta': [[3.2], [3.2]], 'test_data': [[5.5], [4.4]]}
            ],
            scipy_arg_fn=lambda alpha, beta: ((np.array(alpha),),
                                              {"scale": 1.0 / np.array(beta)})),
    Fixture(pyro_dist=dist.Beta,
            scipy_dist=sp.beta,
            examples=[
                {'alpha': [2.4], 'beta': [3.6],
                 'test_data': [0.4]},
                {'alpha': [[2.4, 2.4], [3.6, 3.6]], 'beta': [[2.5, 2.5], [2.5, 2.5]],
                 'test_data': [[[5.5, 4.4], [5.5, 4.4]]]},
                {'alpha': [[2.4], [3.7]], 'beta': [[3.6], [2.5]],
                 'test_data': [[0.4], [0.6]]}
            ],
            scipy_arg_fn=lambda alpha, beta: ((np.array(alpha), np.array(beta)), {})),
    Fixture(pyro_dist=NaiveBeta,
            scipy_dist=sp.beta,
            examples=[
                {'alpha': [2.4], 'beta': [3.6],
                 'test_data': [0.4]},
                {'alpha': [[2.4, 2.4], [3.6, 3.6]], 'beta': [[2.5, 2.5], [2.5, 2.5]],
                 'test_data': [[[5.5, 4.4], [5.5, 4.4]]]},
                {'alpha': [[2.4], [3.7]], 'beta': [[3.6], [2.5]],
                 'test_data': [[0.4], [0.6]]}
            ],
            scipy_arg_fn=lambda alpha, beta: ((np.array(alpha), np.array(beta)), {})),
    Fixture(pyro_dist=ShapeAugmentedBeta,
            scipy_dist=sp.beta,
            examples=[
                {'alpha': [2.4], 'beta': [3.6],
                 'test_data': [0.4]},
                {'alpha': [[2.4, 2.4], [3.6, 3.6]], 'beta': [[2.5, 2.5], [2.5, 2.5]],
                 'test_data': [[[5.5, 4.4], [5.5, 4.4]]]},
                {'alpha': [[2.4], [3.7]], 'beta': [[3.6], [2.5]],
                 'test_data': [[0.4], [0.6]]}
            ],
            scipy_arg_fn=lambda alpha, beta: ((np.array(alpha), np.array(beta)), {})),
    Fixture(pyro_dist=dist.LogNormal,
            scipy_dist=sp.lognorm,
            examples=[
                {'mu': [1.4], 'sigma': [0.4],
                 'test_data': [5.5]},
                {'mu': [1.4], 'sigma': [0.4],
                 'test_data': [[5.5]]},
                {'mu': [[1.4, 0.4], [1.4, 0.4]], 'sigma': [[2.6, 0.5], [2.6, 0.5]],
                 'test_data': [[5.5, 6.4], [5.5, 6.4]]},
                {'mu': [[1.4], [0.4]], 'sigma': [[2.6], [0.5]],
                 'test_data': [[5.5], [6.4]]}
            ],
            scipy_arg_fn=lambda mu, sigma: ((np.array(sigma),), {"scale": np.exp(np.array(mu))})),
    Fixture(pyro_dist=dist.Normal,
            scipy_dist=sp.norm,
            examples=[
                {'mu': [2.0], 'sigma': [4.0],
                 'test_data': [2.0]},
                {'mu': [[2.0]], 'sigma': [[4.0]],
                 'test_data': [[2.0]]},
                {'mu': [[[2.0]]], 'sigma': [[[4.0]]],
                 'test_data': [[[2.0]]]},
                {'mu': [2.0, 50.0], 'sigma': [4.0, 100.0],
                 'test_data': [[2.0, 50.0], [2.0, 50.0]]},
            ],
            scipy_arg_fn=lambda mu, sigma: ((), {"loc": np.array(mu), "scale": np.array(sigma)}),
            prec=0.07,
            min_samples=50000),
    Fixture(pyro_dist=dist.MultivariateNormal,
            scipy_dist=sp.multivariate_normal,
            examples=[
                {'loc': [2.0, 1.0], 'covariance_matrix': [[1.0, 0.5], [0.5, 1.0]],
                 'test_data': [[2.0, 1.0], [9.0, 3.4]]},
            ],
            # This hack seems to be the best option right now, as 'sigma' is not handled well by get_scipy_batch_logpdf
            scipy_arg_fn=lambda loc, covariance_matrix=None:
                ((), {"mean": np.array(loc), "cov": np.array([[1.0, 0.5], [0.5, 1.0]])}),
            prec=0.01,
            min_samples=500000),
    Fixture(pyro_dist=dist.SparseMultivariateNormal,
            scipy_dist=sp.multivariate_normal,
            examples=[
                {'loc': [2.0, 1.0], 'D_term': [0.5, 0.5], 'W_term': [[1.0, 0.5]],
                 'test_data': [[2.0, 1.0], [9.0, 3.4]]},
            ],
            scipy_arg_fn=lambda loc, D_term=None, W_term=None:
                ((), {"mean": np.array(loc), "cov": np.array([[1.5, 0.5], [0.5, 0.75]])}),
            prec=0.01,
            min_samples=500000),
    Fixture(pyro_dist=dist.Dirichlet,
            scipy_dist=sp.dirichlet,
            examples=[
                {'alpha': [2.4, 3, 6],
                 'test_data': [0.2, 0.45, 0.35]},
                {'alpha': [2.4, 3, 6],
                 'test_data': [[0.2, 0.45, 0.35], [0.2, 0.45, 0.35]]},
                {'alpha': [[2.4, 3, 6], [3.2, 1.2, 0.4]],
                 'test_data': [[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]]}
            ],
            scipy_arg_fn=lambda alpha: ((alpha,), {})),
    Fixture(pyro_dist=NaiveDirichlet,
            scipy_dist=sp.dirichlet,
            examples=[
                {'alpha': [2.4, 3, 6],
                 'test_data': [0.2, 0.45, 0.35]},
                {'alpha': [2.4, 3, 6],
                 'test_data': [[0.2, 0.45, 0.35], [0.2, 0.45, 0.35]]},
                {'alpha': [[2.4, 3, 6], [3.2, 1.2, 0.4]],
                 'test_data': [[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]]}
            ],
            scipy_arg_fn=lambda alpha: ((alpha,), {})),
    Fixture(pyro_dist=ShapeAugmentedDirichlet,
            scipy_dist=sp.dirichlet,
            examples=[
                {'alpha': [2.4, 3, 6],
                 'test_data': [0.2, 0.45, 0.35]},
                {'alpha': [2.4, 3, 6],
                 'test_data': [[0.2, 0.45, 0.35], [0.2, 0.45, 0.35]]},
                {'alpha': [[2.4, 3, 6], [3.2, 1.2, 0.4]],
                 'test_data': [[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]]}
            ],
            scipy_arg_fn=lambda alpha: ((alpha,), {})),
    Fixture(pyro_dist=dist.Cauchy,
            scipy_dist=sp.cauchy,
            examples=[
                {'mu': [0.5], 'gamma': [1.2],
                 'test_data': [1.0]},
                {'mu': [0.5, 0.5], 'gamma': [1.2, 1.2],
                 'test_data': [[1.0, 1.0], [1.0, 1.0]]},
                {'mu': [[0.5], [0.3]], 'gamma': [[1.2], [1.0]],
                 'test_data': [[0.4], [0.35]]}
            ],
            scipy_arg_fn=lambda mu, gamma: ((), {"loc": np.array(mu), "scale": np.array(gamma)})),
]

discrete_dists = [
    Fixture(pyro_dist=dist.Multinomial,
            scipy_dist=sp.multinomial,
            examples=[
                {'ps': [0.1, 0.6, 0.3],
                 'test_data': [0, 1, 0]},
                {'ps': [0.1, 0.6, 0.3], 'n': [8],
                 'test_data': [2, 4, 2]},
                {'ps': [0.1, 0.6, 0.3], 'n': [8],
                 'test_data': [[2, 4, 2], [2, 4, 2]]},
                {'ps': [[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]], 'n': [[8], [8]],
                 'test_data': [[2, 4, 2], [1, 4, 3]]}
            ],
            scipy_arg_fn=lambda ps, n=[1]: ((n[0], np.array(ps)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.Bernoulli,
            scipy_dist=sp.bernoulli,
            examples=[
                {'ps': [0.25],
                 'test_data': [1]},
                {'ps': [0.25, 0.25],
                 'test_data': [[[0, 1]], [[1, 0]], [[0, 0]]]},
                {'logits': [math.log(p / (1 - p)) for p in (0.25, 0.25)],
                 'test_data': [[[0, 1]], [[1, 0]], [[0, 0]]]},
                # for now, avoid tests on infinite logits
                # {'logits': [-float('inf'), 0],
                #  'test_data': [[0, 1], [0, 1], [0, 1]]},
                {'logits': [[math.log(p / (1 - p)) for p in (0.25, 0.25)],
                            [math.log(p / (1 - p)) for p in (0.3, 0.3)]],
                 'test_data': [[1, 1], [0, 0]]},
                {'ps': [[0.25, 0.25], [0.3, 0.3]],
                 'test_data': [[1, 1], [0, 0]]}
            ],
            # for now, avoid tests on infinite logits
            # test_data_indices=[0, 1, 2, 3],
            batch_data_indices=[-1, -2],
            scipy_arg_fn=lambda **kwargs: ((), {'p': kwargs['ps']}),
            prec=0.01,
            min_samples=10000,
            is_discrete=True,
            expected_support_non_vec=[[0], [1]],
            expected_support=[[[0, 0], [0, 0]], [[1, 1], [1, 1]]]),
    Fixture(pyro_dist=dist.Binomial,
            scipy_dist=sp.binom,
            examples=[
                {'ps': [0.6], 'n': 8,
                 'test_data': [4]},
                {'ps': [0.3], 'n': 8,
                 'test_data': [[2], [4]]},
                {'ps': [[0.2], [0.4]], 'n': 8,
                 'test_data': [[4], [3]]}
            ],
            scipy_arg_fn=lambda ps, n: ((n, ps[0]), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.Categorical,
            scipy_dist=sp.multinomial,
            examples=[
                {'ps': [0.1, 0.6, 0.3],
                 'test_data': [2]},
                {'logits': list(map(math.log, [0.1, 0.6, 0.3])),
                 'test_data': [2]},
                {'logits': [list(map(math.log, [0.1, 0.6, 0.3])),
                            list(map(math.log, [0.2, 0.4, 0.4]))],
                 'test_data': [2, 0]},
                {'ps': [[0.1, 0.6, 0.3],
                        [0.2, 0.4, 0.4]],
                 'test_data': [2, 0]}
            ],
            test_data_indices=[0, 1, 2],
            batch_data_indices=[-1, -2],
            scipy_arg_fn=None,
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.OneHotCategorical,
            scipy_dist=sp.multinomial,
            examples=[
                {'ps': [0.1, 0.6, 0.3],
                 'test_data': [0, 0, 1]},
                {'logits': list(map(math.log, [0.1, 0.6, 0.3])),
                 'test_data': [0, 0, 1]},
                {'logits': [list(map(math.log, [0.1, 0.6, 0.3])),
                            list(map(math.log, [0.2, 0.4, 0.4]))],
                 'test_data': [[0, 0, 1], [1, 0, 0]]},
                {'ps': [[0.1, 0.6, 0.3],
                        [0.2, 0.4, 0.4]],
                 'test_data': [[0, 0, 1], [1, 0, 0]]}
            ],
            test_data_indices=[0, 1, 2],
            batch_data_indices=[-1, -2],
            scipy_arg_fn=lambda ps: ((1, np.array(ps)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.Poisson,
            scipy_dist=sp.poisson,
            examples=[
                {'lam': [2.0],
                 'test_data': [0]},
                {'lam': [3.0],
                 'test_data': [1]},
                {'lam': [6.0],
                 'test_data': [4]},
                {'lam': [2.0, 3.0, 6.0],
                 'test_data': [[0, 1, 4], [0, 1, 4]]},
                {'lam': [[2.0], [3.0], [6.0]],
                 'test_data': [[0], [1], [4]]}
            ],
            scipy_arg_fn=lambda lam: ((np.array(lam),), {}),
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
