from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.distributions import AffineTransform, ExpTransform, ComposeTransform

import pyro.distributions as dist
from pyro.distributions import LogNormal
from pyro.distributions import TransformedDistribution
from pyro.distributions.util import torch_ones_like, torch_zeros_like
from tests.common import assert_equal
from tests.distributions.dist_fixture import Fixture


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
            'mu': [1.4, 0.4, 0.4],
            'sigma': [1.2, 0.5, 0.5],
            'test_data': [[5.5, 6.4, 6.4], [0.5, 0.4, 0.4]],
        },
        'min_samples': 500000,
    },
    {
        'example': {'mu': [1.4], 'sigma': [1.2], 'test_data': [[5.5], [6.4]]},
        'min_samples': 1000000,
    },
]))


def unwrap_variable(x):
    return x.data.cpu().numpy()


def AffineExp(affine_b, affine_a):
    affine_transform = AffineTransform(loc=affine_a, scale=affine_b)
    exp_transform = ExpTransform()
    return ComposeTransform([affine_transform, exp_transform])


def get_transformed_dist(distribution, affine_a, affine_b):
    return TransformedDistribution(distribution, [AffineExp(affine_b, affine_a)])


@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_mean_and_var(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    mu_z = torch_zeros_like(mu_lognorm)
    sigma_z = torch_ones_like(sigma_lognorm)
    normal_dist = dist.Normal(mu_z, sigma_z)
    trans_dist = get_transformed_dist(normal_dist, mu_lognorm, sigma_lognorm)
    torch_samples = trans_dist.sample(sample_shape=torch.Size((lognormal.get_num_samples(0),)))
    torch_mean = torch.mean(torch_samples, 0)
    torch_std = torch.std(torch_samples, 0)
    analytic_mean = lognormal.pyro_dist.analytic_mean(**dist_params)
    analytic_std = lognormal.pyro_dist.analytic_var(**dist_params) ** 0.5
    precision = analytic_mean.max().data[0] * 0.05
    assert_equal(torch_mean, analytic_mean, prec=precision)
    assert_equal(torch_std, analytic_std, prec=precision)


@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_log_pdf(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    mu_z = torch_zeros_like(mu_lognorm)
    sigma_z = torch_ones_like(sigma_lognorm)
    normal_dist = dist.Normal(mu_z, sigma_z)
    trans_dist = get_transformed_dist(normal_dist, mu_lognorm, sigma_lognorm)
    test_data = lognormal.get_test_data(0)
    log_px_torch = trans_dist.log_prob(test_data).sum().data[0]
    log_px_np = sp.lognorm.logpdf(
        test_data.data.cpu().numpy(),
        sigma_lognorm.data.cpu().numpy(),
        scale=np.exp(mu_lognorm.data.cpu().numpy())).sum()
    assert_equal(log_px_torch, log_px_np, prec=1e-4)


@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_log_prob(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    mu_z = torch_zeros_like(mu_lognorm)
    sigma_z = torch_ones_like(sigma_lognorm)
    normal_dist = dist.Normal(mu_z, sigma_z)
    trans_dist = get_transformed_dist(normal_dist, mu_lognorm, sigma_lognorm)
    test_data = lognormal.get_test_data(0)
    log_px_torch = trans_dist.log_prob(test_data).data.cpu().numpy()
    log_px_np = sp.lognorm.logpdf(
        test_data.data.cpu().numpy(),
        sigma_lognorm.data.cpu().numpy(),
        scale=np.exp(mu_lognorm.data.cpu().numpy()))
    assert_equal(log_px_torch, log_px_np, prec=1e-4)


@pytest.mark.parametrize('lognormal', EXAMPLES)
def test_shape(lognormal):
    dist_params = lognormal.get_dist_params(0)
    mu_lognorm = dist_params['mu']
    sigma_lognorm = dist_params['sigma']
    normal_dist = dist.Normal(**dist_params)
    trans_dist = get_transformed_dist(normal_dist, mu_lognorm, sigma_lognorm)
    assert_equal(trans_dist.sample().size(), trans_dist.shape())
