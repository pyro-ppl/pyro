from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from pyro.distributions import Normal, Gamma, Summary, NIGNormalRegressionSummary
from numpy.testing import assert_almost_equal

def test_nig_smoke():
    # test conversion between forms for NIGNormal
    
    summary = NIGNormalRegressionSummary(torch.tensor([2., 1.]), 
                                    torch.tensor([[1., 0.],[0., 1.]]), 3., 1.)

    features = torch.rand((5,2))
    obs = torch.rand((5,1))
    summary.update(obs, features)

    NIGNormalRegressionSummary.convert_to_canonical_form(summary.precision_times_mean,
                            summary.precision, summary.shape, summary.reparameterized_scale)

def test_nig_prior():
    true_mean = torch.tensor([2., 1.])
    true_cov = torch.tensor([[5., 2.],[2., 5.]])
    true_shape = 3.
    true_scale = 1.
    summary = NIGNormalRegressionSummary(true_mean, true_cov, true_shape, true_scale)
    mean, cov, shape, scale = NIGNormalRegressionSummary.convert_to_canonical_form(summary.precision_times_mean,
                                        summary.precision, summary.shape, summary.reparameterized_scale)

    assert torch.all(mean.eq(true_mean))
    assert torch.all(cov.eq(true_cov))
    assert_almost_equal(shape, true_shape, decimal=7)
    assert_almost_equal(scale, true_scale, decimal=7)

def test_nig_asymptotics():
    # test the likelihood being correct
    # include conversions between forms
    weights = torch.tensor([2., 1.])
    variance = 10.
    noise = Normal(0., np.sqrt(variance))
    features = torch.rand((500,2))
    obs = features.matmul(weights).unsqueeze(-1) + noise.sample(sample_shape=torch.Size([500, 1]))

    summary = NIGNormalRegressionSummary(torch.tensor([0., 0.]), 
                                    torch.tensor([[3., 0.],[0., 3.]]), 1.1, 10.)
    summary.update(obs, features)
    mean, cov, shape, scale = NIGNormalRegressionSummary.convert_to_canonical_form(summary.precision_times_mean,
                                        summary.precision, summary.shape, summary.reparameterized_scale)

    print(Gamma(shape, 1./scale).log_prob(torch.tensor(1./variance)))
    print(1./Gamma(shape, 1./scale).mean)
    print(Gamma(shape, 1./scale).variance)
    assert torch.exp(Gamma(shape, 1./scale).log_prob(torch.tensor(1./variance))) > 0.9
