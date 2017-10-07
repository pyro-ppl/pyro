import math

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from pyro.distributions.transformed_distribution import AffineExp, TransformedDistribution
from tests.common import assert_equal
from tests.distributions.dist_fixture import Fixture


"""
If X is a lognormal RV, then it can be expressed as
X = e^(mu + sigma * Z),
where Z is drawn from a standard normal

This is the same as sampling Z from a standard normal, and doing an AffineExp transformation:
Y = e^(a * Z + b),
where a = sigma
and, b = mu
"""


@pytest.fixture()
def lognormal():
    return Fixture(dist.lognormal,
                   sp.lognorm,
                   [(1.4, 0.4)],
                   [5.5],
                   lambda mean, sigma: ((sigma,), {"scale": math.exp(mean)}),
                   prec=0.1,
                   min_samples=20000)


def unwrap_variable(x):
    return x.data.cpu().numpy()


def get_transformed_dist(distribution, affine_a, affine_b):
    bijector = AffineExp(affine_a, affine_b)
    return TransformedDistribution(distribution, bijector)


def test_mean_and_var_on_transformed_distribution(lognormal):
    mu_z = Variable(torch.zeros(1))
    sigma_z = Variable(torch.ones(1))
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm, sigma_lognorm = lognormal.get_dist_params(0)
    trans_dist = get_transformed_dist(dist.diagnormal, sigma_lognorm, mu_lognorm)
    torch_samples = [trans_dist.sample(mu_z, sigma_z).data.cpu().numpy()
                     for _ in range(lognormal.get_num_samples(0))]
    torch_mean = np.mean(torch_samples)
    torch_var = np.var(torch_samples)
    analytic_mean = unwrap_variable(lognormal.pyro_dist.analytic_mean(*dist_params))[0]
    analytic_var = unwrap_variable(lognormal.pyro_dist.analytic_var(*dist_params))[0]
    assert_equal(torch_mean, analytic_mean, prec=0.1)
    assert_equal(torch_var, analytic_var, prec=0.1)


def test_log_pdf_on_transformed_distribution(lognormal):
    mu_z = Variable(torch.zeros(1))
    sigma_z = Variable(torch.ones(1))
    mu_lognorm, sigma_lognorm = lognormal.get_dist_params(0)
    trans_dist = get_transformed_dist(dist.diagnormal, sigma_lognorm, mu_lognorm)
    test_data = lognormal.get_test_data(0)
    log_px_torch = trans_dist.log_pdf(test_data, mu_z, sigma_z).data[0]
    log_px_np = sp.lognorm.logpdf(
        test_data.data.cpu().numpy(),
        sigma_lognorm.data.cpu().numpy(),
        scale=np.exp(mu_lognorm.data.cpu().numpy()))[0]
    assert_equal(log_px_torch, log_px_np, prec=1e-4)
