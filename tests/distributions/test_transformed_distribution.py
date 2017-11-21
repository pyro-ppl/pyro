from __future__ import absolute_import, division, print_function

import numbers

import numpy as np
import pytest
import scipy.stats as sp
import torch

import pyro.distributions as dist
from pyro.distributions import LogNormal
from pyro.distributions.transformed_distribution import Bijector, TransformedDistribution
from pyro.distributions.util import torch_ones_like, torch_zeros_like
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

    def batch_log_det_jacobian(self, y, *args, **kwargs):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        return (torch.log(torch.abs(self.a)) + torch.log(y)).sum(-1).unsqueeze(-1)


def make_lognormal(kwargs):
    kwargs['examples'] = [kwargs.pop('example')]
    return Fixture(pyro_dist=(dist.lognormal, LogNormal),
                   scipy_dist=sp.lognorm,
                   scipy_arg_fn=lambda mu, sigma: ((np.array(sigma),),
                                                   {"scale": np.exp(np.array(mu))}),
                   **kwargs)


EXAMPLES = list(map(make_lognormal, [
    {
        'example': {'mu': [1.4], 'sigma': [0.4], 'test_data': [5.5]},
    },
    {
        'example': {'mu': [1.4], 'sigma': [0.4], 'test_data': [[5.5]]},
    },
    {
        'example': {
            'mu': [[1.4, 0.4, 0.4], [1.4, 0.4, 0.6]],
            'sigma': [[2.6, 0.5, 0.5], [2.6, 0.5, 0.5]],
            'test_data': [[5.5, 6.4, 6.4], [0.5, 0.4, 0.4]],
        },
        'min_samples': 10000,
    },
    {
        'example': {'mu': [[1.4], [0.4]], 'sigma': [[2.6], [0.5]], 'test_data': [[5.5], [6.4]]},
        'min_samples': 10000,
    },
]))


def unwrap_variable(x):
    return x.data.cpu().numpy()


def get_transformed_dist(distribution, affine_a, affine_b):
    bijector = AffineExp(affine_a, affine_b)
    return TransformedDistribution(distribution, bijector)


@pytest.mark.xfail(reason='https://github.com/uber/pyro/issues/293')
@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_mean_and_var(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    mu_z = torch_zeros_like(mu_lognorm)
    sigma_z = torch_ones_like(sigma_lognorm)
    trans_dist = get_transformed_dist(dist.normal, sigma_lognorm, mu_lognorm)
    torch_samples = np.zeros([lognormal.get_num_samples(0)] + list(trans_dist.batch_shape(None, mu_z, sigma_z)))
    for i in range(len(torch_samples)):
        torch_samples[i] = trans_dist.sample(mu_z, sigma_z).data.cpu().numpy()
    torch_mean = np.mean(torch_samples, axis=0)
    torch_std = np.var(torch_samples, axis=0) ** 0.5
    analytic_mean = unwrap_variable(lognormal.pyro_dist.analytic_mean(**dist_params))
    analytic_std = unwrap_variable(lognormal.pyro_dist.analytic_var(**dist_params)) ** 0.5
    if isinstance(torch_mean, numbers.Number):
        analytic_mean = analytic_mean[0]
        analytic_std = analytic_std[0]
    assert_equal(torch_mean, analytic_mean, prec=0.1)
    assert_equal(torch_std, analytic_std, prec=0.1)


@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_log_pdf(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    mu_z = torch_zeros_like(mu_lognorm)
    sigma_z = torch_ones_like(sigma_lognorm)
    trans_dist = get_transformed_dist(dist.normal, sigma_lognorm, mu_lognorm)
    test_data = lognormal.get_test_data(0)
    log_px_torch = trans_dist.log_pdf(test_data, mu_z, sigma_z).data[0]
    log_px_np = sp.lognorm.logpdf(
        test_data.data.cpu().numpy(),
        sigma_lognorm.data.cpu().numpy(),
        scale=np.exp(mu_lognorm.data.cpu().numpy())).sum()
    assert_equal(log_px_torch, log_px_np, prec=1e-4)


@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_batch_log_pdf(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    mu_z = torch_zeros_like(mu_lognorm)
    sigma_z = torch_ones_like(sigma_lognorm)
    trans_dist = get_transformed_dist(dist.normal, sigma_lognorm, mu_lognorm)
    test_data = lognormal.get_test_data(0)
    log_px_torch = trans_dist.batch_log_pdf(test_data, mu_z, sigma_z).data.cpu().numpy()
    log_px_np = sp.lognorm.logpdf(
        test_data.data.cpu().numpy(),
        sigma_lognorm.data.cpu().numpy(),
        scale=np.exp(mu_lognorm.data.cpu().numpy())).sum(-1, keepdims=True)
    assert_equal(log_px_torch, log_px_np, prec=1e-4)


@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_shape(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    d = get_transformed_dist(dist.normal, sigma_lognorm, mu_lognorm)
    assert_equal(d.sample(**dist_params).size(), d.shape(**dist_params))
