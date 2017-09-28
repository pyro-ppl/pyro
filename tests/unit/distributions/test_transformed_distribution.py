import math

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from pyro.distributions.transformed_distribution import AffineExp, TransformedDistribution
from tests.common import assert_equal
from tests.unit.distributions.dist_fixture import Fixture

pytestmark = pytest.mark.init(rng_seed=123)


@pytest.fixture()
def distribution():
    return Fixture(dist.lognormal,
                   sp.lognorm,
                   [(1.4, 0.4)],
                   [5.5],
                   lambda mean, sigma: ((sigma,), {"scale": math.exp(mean)}),
                   prec=0.1,
                   min_samples=20000)


def unwrap_variable(x):
    return x.data.cpu().numpy()


def get_transformed_dist(distribution):
    dist_params = distribution.get_dist_params(0)
    mu, sigma = dist_params
    bijector = AffineExp(sigma, mu)
    return TransformedDistribution(dist.diagnormal, bijector)


def test_mean_and_var_on_transformed_distribution(distribution):
    zero = Variable(torch.zeros(1))
    one = Variable(torch.ones(1))
    dist_params = distribution.get_dist_params(0)
    trans_dist = get_transformed_dist(distribution)
    torch_samples = [trans_dist.sample(zero, one).data.cpu().numpy()
                     for _ in range(distribution.get_num_samples(0))]
    torch_mean = np.mean(torch_samples)
    torch_var = np.var(torch_samples)
    analytic_mean = unwrap_variable(distribution.pyro_dist.analytic_mean(*dist_params))[0]
    analytic_var = unwrap_variable(distribution.pyro_dist.analytic_var(*dist_params))[0]
    assert_equal(torch_mean, analytic_mean, prec=0.1)
    assert_equal(torch_var, analytic_var, prec=0.1)


def test_log_pdf_on_transformed_distribution(distribution):
    zero = Variable(torch.zeros(1))
    one = Variable(torch.ones(1))
    trans_dist = get_transformed_dist(distribution)
    test_data = distribution.get_test_data(0)
    log_px_torch = trans_dist.log_pdf(test_data, zero, one).data[0]
    mu, sigma = distribution.get_dist_params(0)
    log_px_np = sp.lognorm.logpdf(
        test_data.data.cpu().numpy(),
        sigma.data.cpu().numpy(),
        scale=np.exp(mu.data.cpu().numpy()))[0]
    assert_equal(log_px_torch, log_px_np, prec=1e-4)
