import torch
from torch.testing import assert_allclose
import numpy as np
from pyro.distributions import Bernoulli, Beta, BetaBernoulliSummary, Gamma, Normal, NIGNormalRegressionSummary


def test_betabern_smoke():
    summary = BetaBernoulliSummary(2., 4.)

    obs = torch.rand((1, 1))
    batch_obs = torch.rand((5, 1))

    summary.update(obs)
    summary.update(batch_obs)


def test_betabern_asymptotics():
    summary = BetaBernoulliSummary(2., 4.)
    obs = Bernoulli(probs=0.3).sample(sample_shape=torch.Size([100, 1]))

    summary.update(obs)
    assert_allclose(Beta(summary.alpha, summary.beta).mean, 0.3, rtol=0.0, atol=0.05)
    assert_allclose(Beta(summary.alpha, summary.beta).variance, 0.0, rtol=0.0, atol=0.1)


def test_nignorm_single_smoke():
    summary = NIGNormalRegressionSummary(torch.tensor([5.]), torch.tensor([[6.]]), 3., 1.)

    features = torch.rand((1, 1))
    obs = torch.rand((1, 1))
    summary.update(obs, features)

    NIGNormalRegressionSummary.convert_to_canonical_form(summary.precision_times_mean,
                                                         summary.precision,
                                                         summary.shape,
                                                         summary.reparameterized_rate)


def test_nignorm_multi_smoke():
    # test conversion between forms for NIGNormal
    summary = NIGNormalRegressionSummary(torch.tensor([2., 1.]),
                                         torch.tensor([[1., 0.], [0., 1.]]), 3., 1.)

    features = torch.rand((5, 2))
    obs = torch.rand((5, 1))
    summary.update(obs, features)

    NIGNormalRegressionSummary.convert_to_canonical_form(summary.precision_times_mean,
                                                         summary.precision,
                                                         summary.shape,
                                                         summary.reparameterized_rate)


def test_nignorm_prior():
    true_mean = torch.tensor([2., 1.])
    true_cov = torch.tensor([[5., 2.], [2., 5.]])
    true_shape = 3.
    true_rate = 1.
    summary = NIGNormalRegressionSummary(true_mean, true_cov, true_shape, true_rate)
    mean, cov, shape, rate = NIGNormalRegressionSummary.convert_to_canonical_form(summary.precision_times_mean,
                                                                                  summary.precision,
                                                                                  summary.shape,
                                                                                  summary.reparameterized_rate)

    assert_allclose(mean, true_mean)
    assert_allclose(cov, true_cov)
    assert_allclose(shape, true_shape)
    assert_allclose(rate, true_rate)


def test_nignorm_asymptotics():
    # test the likelihood being correct
    # include conversions between forms
    weights = torch.tensor([2., 1.])
    variance = 10.
    noise = Normal(0., np.sqrt(variance))
    features = torch.rand((10000, 2))
    obs = features.matmul(weights).unsqueeze(-1) + noise.sample(sample_shape=torch.Size([10000, 1]))

    summary = NIGNormalRegressionSummary(torch.tensor([0., 0.]),
                                         torch.tensor([[3., 0.], [0., 3.]]), 1.1, 10.)
    summary.update(obs, features)
    mean, cov, shape, rate = NIGNormalRegressionSummary.convert_to_canonical_form(summary.precision_times_mean,
                                                                                  summary.precision,
                                                                                  summary.shape,
                                                                                  summary.reparameterized_rate)

    assert_allclose(mean, weights, rtol=0.0, atol=0.1)
    assert_allclose(cov, torch.zeros((2, 2)), rtol=0.0, atol=0.1)
    assert_allclose(1./Gamma(shape, rate).mean, variance, rtol=0.0, atol=0.1)
    assert_allclose(Gamma(shape, rate).variance, 0., rtol=0.0, atol=0.1)
