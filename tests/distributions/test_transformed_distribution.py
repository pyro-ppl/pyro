from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from pyro.distributions import LogNormal
from pyro.distributions.transformed_distribution import Bijector, TransformedDistribution
from tests.common import assert_equal
from tests.distributions.dist_fixture import Fixture


class AffineExp(Bijector):
    """
    :param a_init: a
    :param b_init: b

    `y = exp(ax + b)`

    If X is a lognormal RV, then it can be expressed as
    X = e^(mu + sigma * Z),
    where Z is drawn from a standard normal

    This is the same as sampling Z from a standard normal, and doing an AffineExp transformation:
    Y = e^(a * Z + b),
    where a = sigma
    """

    def __init__(self, a_init, b_init):
        """
        Constructor for univariate affine bijector followed by exp
        """
        super(AffineExp, self).__init__()
        self.a = a_init
        self.b = b_init

    def __call__(self, x, *args, **kwargs):
        """
        Invoke bijection x=>y
        """
        y = self.a * x + self.b
        return torch.exp(y)

    def inverse(self, y, *args, **kwargs):
        """
        Invert y => x
        """
        x = (torch.log(y) - self.b) / self.a
        return x

    def log_det_jacobian(self, y, *args, **kwargs):
        """
        Calculates the determinant of the log jacobian
        """
        return torch.log(torch.abs(self.a)) + torch.log(y)


@pytest.fixture()
def lognormal():
    return Fixture(pyro_dist=(dist.lognormal, LogNormal),
                   scipy_dist=sp.lognorm,
                   examples=[
                       {'mu': [1.4], 'sigma': [0.4], 'test_data': [5.5]},
                   ],
                   scipy_arg_fn=lambda mu, sigma: ((np.array(sigma),), {"scale": np.exp(np.array(mu))}))


def unwrap_variable(x):
    return x.data.cpu().numpy()


def get_transformed_dist(distribution, affine_a, affine_b):
    bijector = AffineExp(affine_a, affine_b)
    return TransformedDistribution(distribution, bijector)


def test_mean_and_var_on_transformed_distribution(lognormal):
    mu_z = Variable(torch.zeros(1))
    sigma_z = Variable(torch.ones(1))
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    trans_dist = get_transformed_dist(dist.normal, sigma_lognorm, mu_lognorm)
    torch_samples = [trans_dist.sample(mu_z, sigma_z).data.cpu().numpy()
                     for _ in range(lognormal.get_num_samples(0))]
    torch_mean = np.mean(torch_samples)
    torch_var = np.var(torch_samples)
    analytic_mean = unwrap_variable(lognormal.pyro_dist.analytic_mean(**dist_params))[0]
    analytic_var = unwrap_variable(lognormal.pyro_dist.analytic_var(**dist_params))[0]
    assert_equal(torch_mean, analytic_mean, prec=0.1)
    assert_equal(torch_var, analytic_var, prec=0.1)


def test_log_pdf_on_transformed_distribution(lognormal):
    mu_z = Variable(torch.zeros(1))
    sigma_z = Variable(torch.ones(1))
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    trans_dist = get_transformed_dist(dist.normal, sigma_lognorm, mu_lognorm)
    test_data = lognormal.get_test_data(0)
    log_px_torch = trans_dist.log_pdf(test_data, mu_z, sigma_z).data[0]
    log_px_np = sp.lognorm.logpdf(
        test_data.data.cpu().numpy(),
        sigma_lognorm.data.cpu().numpy(),
        scale=np.exp(mu_lognorm.data.cpu().numpy()))[0]
    assert_equal(log_px_torch, log_px_np, prec=1e-4)
