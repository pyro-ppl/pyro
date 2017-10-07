import math
import os

import numpy as np
import pytest
import scipy.stats as sp

import pyro.distributions as dist
from tests.distributions.dist_fixture import Fixture
from tests.common import RESOURCE_DIR

continuous_dists = [
    Fixture(pyro_dist=dist.uniform,
            scipy_dist=sp.uniform,
            dist_params=[(2, 2.5), (-3, 0), (0, 1)],
            test_data=[2.2, -2, 0.7],
            scipy_arg_fn=lambda a, b: ((), {"loc": a, "scale": b - a})),
    Fixture(pyro_dist=dist.exponential,
            scipy_dist=sp.expon,
            dist_params=[(2.4,), (1.4,)],
            test_data=[5.5, 3.2],
            scipy_arg_fn=lambda lam: ((), {"scale": 1.0 / lam})),
    Fixture(pyro_dist=dist.gamma,
            scipy_dist=sp.gamma,
            dist_params=[(2.4, 2.4), (3.2, 3.2)],
            test_data=[5.5, 4.4],
            scipy_arg_fn=lambda a, b: (([a]), {"scale": 1.0 / b})),
    Fixture(pyro_dist=dist.beta,
            scipy_dist=sp.beta,
            dist_params=[(2.4, 3.7), (3.6, 2.5)],
            test_data=[0.4, 0.6],
            scipy_arg_fn=lambda a, b: ((a, b), {})),
    Fixture(pyro_dist=dist.normalchol,
            scipy_dist=sp.multivariate_normal,
            dist_params=[([1.0, 1.0], [[2.0, 0.0], [1.0, 3.0]])],
            test_data=[(0.4, 0.6)],
            scipy_arg_fn=lambda mean, L: ((), {"mean": np.array(mean),
                                               "cov": np.matmul(np.array(L), np.array(L).T)}),
            min_samples=100000),
    Fixture(pyro_dist=dist.diagnormal,
            scipy_dist=sp.multivariate_normal,
            dist_params=[(2.0, 4.0), (50.0, 100.0)],
            test_data=[2.0, 50.0],
            scipy_arg_fn=lambda mean, sigma: ((), {"mean": mean, "cov": sigma ** 2}),
            min_samples=50000),
    Fixture(pyro_dist=dist.lognormal,
            scipy_dist=sp.lognorm,
            dist_params=[(1.4, 0.4), (2.6, 0.5)],
            test_data=[5.5, 6.4],
            scipy_arg_fn=lambda mean, sigma: ((sigma,), {"scale": math.exp(mean)})),
    Fixture(pyro_dist=dist.dirichlet,
            scipy_dist=sp.dirichlet,
            dist_params=[([2.4, 3, 6],), ([3.2, 1.2, 0.4],)],
            test_data=[[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]],
            scipy_arg_fn=lambda alpha: ((alpha,), {})),
]

discrete_dists = [
    Fixture(pyro_dist=dist.multinomial,
            scipy_dist=sp.multinomial,
            dist_params=[([0.1, 0.6, 0.3], 8), ([0.2, 0.4, 0.4], 8)],
            test_data=[[2, 4, 2], [1, 4, 3]],
            scipy_arg_fn=lambda ps, n: ((n, np.array(ps)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(pyro_dist=dist.bernoulli,
            scipy_dist=sp.bernoulli,
            dist_params=[([0.25, 0.5, 0.75],), ([0.3, 0.5, 0.7],)],
            test_data=[[1, 0, 1], [1, 0, 0]],
            scipy_arg_fn=lambda ps: ((), {"p": ps}),
            prec=0.01,
            min_samples=10000,
            is_discrete=True,
            expected_support_file=os.path.join(RESOURCE_DIR, 'support_bernoulli.json'),
            expected_support_key='expected'),
    Fixture(pyro_dist=dist.poisson,
            scipy_dist=sp.poisson,
            dist_params=[(2,), (4.5,), (3.,), (5.1,), (6,), (3.2,), (1,)],
            test_data=[0, 1, 2, 4, 4, 1, 2],
            scipy_arg_fn=lambda lam: ((lam,), {}),
            prec=0.08,
            is_discrete=True),
    Fixture(pyro_dist=dist.categorical,
            scipy_dist=sp.multinomial,
            dist_params=[([0.1, 0.6, 0.3],), ([0.2, 0.4, 0.4],)],
            test_data=[[0, 0, 1], [1, 0, 0]],
            scipy_arg_fn=lambda ps: ((1, np.array(ps)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True,
            expected_support_file=os.path.join(RESOURCE_DIR, 'support_categorical.json'),
            expected_support_key='one_hot'),
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


@pytest.fixture(params=[0], ids=lambda x: "obs_" + str(x))
def test_data_idx(request):
    return request.param


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/distributions"):
            if "stage" not in item.keywords:
                item.add_marker(pytest.mark.stage("unit"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
