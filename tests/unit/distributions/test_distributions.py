import math

import numpy as np
import pytest
import scipy.stats as sp
import torch

import pyro.distributions as dist
from tests.common import assert_equal
from tests.unit.distributions.dist_fixture import Fixture

pytestmark = pytest.mark.init(rng_seed=123)

continuous_dists = [
    Fixture(dist.uniform,
            sp.uniform,
            [(2, 2.5), (-3, 0), (0, 1)],
            [2.2, -2, 0.7],
            lambda (a, b): ((), {"loc": a, "scale": b - a})),
    Fixture(dist.exponential,
            sp.expon,
            [(2.4,), (1.4,)],
            [5.5, 3.2],
            lambda (lam, ): ((), {"scale": 1.0 / lam})),
    Fixture(dist.gamma,
            sp.gamma,
            [(2.4, 2.4), (3.2, 3.2)],
            [5.5, 4.4],
            lambda (a, b): (([a]), {"scale": 1.0 / b})),
    Fixture(dist.beta,
            sp.beta,
            [(2.4, 3.7), (3.6, 2.5)],
            [0.4, 0.6],
            lambda (a, b): ((a, b), {})),
    Fixture(dist.normalchol,
            sp.multivariate_normal,
            [([1.0, 1.0], [[2.0, 0.0], [1.0, 3.0]])],
            [(0.4, 0.6)],
            lambda (mean, L): ((), {"mean": np.array(mean),
                                    "cov": np.matmul(np.array(L), np.transpose(np.array(L)))}),
            prec=0.1,
            min_samples=50000),
    Fixture(dist.diagnormal,
            sp.multivariate_normal,
            [(2.0, 4.0), (50.0, 100.0)],
            [2.0, 50.0],
            lambda (mean, sigma): ((), {"mean": mean, "cov": sigma ** 2}),
            prec=0.1,
            min_samples=50000),
    Fixture(dist.lognormal,
            sp.lognorm,
            [(1.4, 0.4), (2.6, 0.5)],
            [5.5, 6.4],
            lambda (mean, sigma): ((sigma,), {"scale": math.exp(mean)}),
            prec=0.1),
    Fixture(dist.dirichlet,
            sp.dirichlet,
            [([2.4, 3, 6],), ([3.2, 1.2, 0.4],)],
            [[0.2, 0.45, 0.35], [0.3, 0.4, 0.3]],
            lambda (alpha, ): ((alpha,), {}),

            )
]

discrete_dists = [
    Fixture(dist.multinomial,
            sp.multinomial,
            [([0.1, 0.6, 0.3], 8), ([0.2, 0.4, 0.4], 8)],
            [[2, 4, 2], [1, 4, 3]],
            lambda (ps, n): ((n, np.array(ps)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True),
    Fixture(dist.bernoulli,
            sp.bernoulli,
            [([0.25, 0.5, 0.75],), ([0.3, 0.5, 0.7],)],
            [[1, 0, 1], [1, 0, 0]],
            lambda (ps, ): ((), {"p": ps}),
            prec=0.01,
            min_samples=10000,
            is_discrete=True,
            expected_support_file='tests/test_data/support_bernoulli.json',
            expected_support_key='expected'),
    Fixture(dist.poisson,
            sp.poisson,
            [(2,), (4.5,), (3.,), (5.1,), (6,), (3.2,), (1,)],
            [0, 1, 2, 4, 4, 1, 2],
            lambda (lam, ): ((lam,), {}),
            prec=0.08,
            is_discrete=True),
    Fixture(dist.categorical,
            sp.multinomial,
            [([0.1, 0.6, 0.3],), ([0.2, 0.4, 0.4],)],
            [[0, 0, 1], [1, 0, 0]],
            lambda (ps, ): ((1, np.array(ps)), {}),
            prec=0.05,
            min_samples=10000,
            is_discrete=True,
            expected_support_file='tests/test_data/support_categorical.json',
            expected_support_key='one_hot')
]


@pytest.fixture(params=continuous_dists + discrete_dists,
                ids=lambda x: x.get_test_distribution_name())
def dist(request):
    return request.param


@pytest.fixture(params=discrete_dists,
                ids=lambda x: x.get_test_distribution_name())
def discrete_dist(request):
    return request.param


@pytest.fixture(params=[0], ids=lambda x: "obs_" + str(x))
def test_data_idx(request):
    return request.param


def unwrap_variable(x):
    return x.data.cpu().numpy()


# Distribution tests

def test_log_pdf(dist, test_data_idx):
    pyro_log_pdf = unwrap_variable(dist.get_pyro_logpdf(test_data_idx))[0]
    scipy_log_pdf = dist.get_scipy_logpdf(test_data_idx)
    assert_equal(pyro_log_pdf, scipy_log_pdf)


def test_batch_log_pdf(dist):
    # TODO (@npradhan) - remove once #144 is resolved
    try:
        logpdf_sum_pyro = unwrap_variable(torch.sum(dist.get_pyro_batch_logpdf()))[0]
    except NotImplementedError:
        return
    logpdf_sum_np = np.sum(dist.get_scipy_batch_logpdf())
    assert_equal(logpdf_sum_pyro, logpdf_sum_np)


def test_mean_and_variance(dist, test_data_idx):
    num_samples = dist.get_num_samples(test_data_idx)
    dist_params = dist.get_dist_params(test_data_idx)
    torch_samples = dist.get_samples(num_samples, *dist_params)
    sample_mean = np.mean(torch_samples, 0)
    sample_var = np.var(torch_samples, 0)
    try:
        analytic_mean = unwrap_variable(dist.pyro_dist.analytic_mean(*dist_params))
        analytic_var = unwrap_variable(dist.pyro_dist.analytic_var(*dist_params))
        assert_equal(sample_mean, analytic_mean, prec=dist.prec)
        assert_equal(sample_var, analytic_var, prec=dist.prec)
    except NotImplementedError:
        pass


def test_support(discrete_dist):
    expected_support = discrete_dist.get_expected_support()
    if not expected_support:
        pytest.skip("Support not tested for distribution")
    actual_support = list(discrete_dist.pyro_dist.support(*discrete_dist.get_dist_params()))
    v = [torch.equal(x.data, y) for x, y in zip(actual_support, expected_support)]
    assert all(v)
