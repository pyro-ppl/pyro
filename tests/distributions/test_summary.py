import pytest
import torch
from torch.testing import assert_allclose
import numpy as np
from pyro.distributions import Bernoulli, Beta, BetaBernoulliSummary, Gamma, Normal, NIGNormalRegressionSummary


def test_betabern_smoke():
    summary = BetaBernoulliSummary(2., 4.)
    multiple_batch_summary = BetaBernoulliSummary(torch.ones(3), torch.ones(3))

    obs = torch.rand((1, 1))
    batch_obs = torch.rand((5, 1))
    multiple_batch_obs1 = torch.rand((3, 5, 1))
    multiple_batch_obs2 = torch.rand((3, 1, 1))

    summary.update(obs)
    summary.update(batch_obs)
    multiple_batch_summary.update(multiple_batch_obs1)
    multiple_batch_summary.update(multiple_batch_obs2)


def test_betabern_asymptotics():
    summary = BetaBernoulliSummary(torch.Tensor([2.]), torch.Tensor([4.]))
    obs = Bernoulli(probs=0.3).sample(sample_shape=torch.Size([100, 1]))

    summary.update(obs)
    assert_allclose(Beta(summary.alpha, summary.beta).mean, 0.3, rtol=0.0, atol=0.05)
    assert_allclose(Beta(summary.alpha, summary.beta).variance, 0.0, rtol=0.0, atol=0.1)


@pytest.mark.parametrize("features_dim", [1, 2])
@pytest.mark.parametrize("obs_dim", [1, 3])
@pytest.mark.parametrize("batch_dim", [1, 4])
def test_nignorm_smoke(features_dim, obs_dim, batch_dim):
    summary_null = NIGNormalRegressionSummary(5., 6., 3., 1.)
    summary_mixed = NIGNormalRegressionSummary(torch.tensor([5.]), 6., torch.tensor(3.), 1.)
    summary_nobatch = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((obs_dim, features_dim)),
                                                 torch.eye(features_dim).expand((obs_dim, features_dim, features_dim)),
                                                 torch.tensor(3.),
                                                 torch.tensor(1.).expand(obs_dim))
    summary_bcast = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((1, features_dim)),
                                               torch.eye(features_dim).expand((1, features_dim, features_dim)),
                                               torch.tensor(3.),
                                               torch.tensor(1.).expand(obs_dim))
    summary_batch = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((batch_dim, obs_dim, features_dim)),
                                               torch.eye(features_dim)
                                               .expand((batch_dim, obs_dim, features_dim, features_dim)),
                                               torch.tensor(3.).expand(obs_dim),
                                               torch.tensor(1.).expand(batch_dim, obs_dim))
    summary_b_bcast = NIGNormalRegressionSummary(torch.zeros(features_dim).expand((batch_dim, 1, features_dim)),
                                                 torch.eye(features_dim)
                                                 .expand((batch_dim, 1, features_dim, features_dim)),
                                                 torch.tensor(3.).expand(1),
                                                 torch.tensor(1.).expand(batch_dim, 1))
    summary_list = [summary_null, summary_mixed, summary_nobatch, summary_bcast, summary_batch, summary_b_bcast]

    features = torch.rand((1, features_dim))
    obs = torch.rand((1, obs_dim))
    features_batch = torch.rand((5, features_dim))
    obs_batch = torch.rand((5, obs_dim))

    for s in summary_list:
        s.update(obs, features)
        s.update(obs_batch, features_batch)
        (s.mean, s.covariance, s.shape, s.rate)

    # TODO: test swapping between obs_dim is not allowed


def test_nignorm_prior():
    true_mean = torch.tensor([2., 1.])
    true_covariance = torch.tensor([[5., 2.], [2., 5.]])
    true_shape = 3.
    true_rate = 1.
    summary = NIGNormalRegressionSummary(true_mean, true_covariance, true_shape, true_rate)

    assert_allclose(summary.mean, true_mean)
    assert_allclose(summary.covariance, true_covariance)
    assert_allclose(summary.shape, true_shape)
    assert_allclose(summary.rate, true_rate)


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

    assert_allclose(summary.mean, weights, rtol=0.0, atol=0.1)
    assert_allclose(summary.covariance, torch.zeros((2, 2)), rtol=0.0, atol=0.1)
    assert_allclose(1./Gamma(summary.shape, summary.rate).mean, variance, rtol=0.0, atol=0.1)
    assert_allclose(Gamma(summary.shape, summary.rate).variance, 0., rtol=0.0, atol=0.1)
