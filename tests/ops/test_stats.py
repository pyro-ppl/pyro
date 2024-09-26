# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
import torch

from pyro.ops.stats import (
    _cummin,
    autocorrelation,
    autocovariance,
    crps_empirical,
    effective_sample_size,
    energy_score_empirical,
    fit_generalized_pareto,
    gelman_rubin,
    hpdi,
    pi,
    quantile,
    resample,
    split_gelman_rubin,
    waic,
    weighed_quantile,
)
from tests.common import assert_close, assert_equal, xfail_if_not_implemented


@pytest.mark.parametrize("replacement", [True, False])
def test_resample(replacement):
    x = torch.empty(10000, 2)
    x[:, 0].normal_(3, 4)
    x[:, 1].normal_(5, 6)

    num_samples = 5000
    y = resample(x, num_samples=num_samples, replacement=replacement)
    z = resample(x.t(), num_samples=num_samples, dim=1, replacement=replacement)
    if not replacement:
        assert_equal(torch.unique(y.reshape(-1)).numel(), y.numel())
        assert_equal(torch.unique(z.reshape(-1)).numel(), z.numel())
    assert_equal(y.shape, torch.Size([num_samples, 2]))
    assert_equal(z.shape, torch.Size([2, num_samples]))
    assert_equal(y.mean(dim=0), torch.tensor([3.0, 5.0]), prec=0.2)
    assert_equal(z.mean(dim=1), torch.tensor([3.0, 5.0]), prec=0.2)
    assert_equal(y.std(dim=0), torch.tensor([4.0, 6.0]), prec=0.2)
    assert_equal(z.std(dim=1), torch.tensor([4.0, 6.0]), prec=0.2)


@pytest.mark.init(rng_seed=3)
def test_quantile():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.rand(2000)
    z = torch.randn(2000)

    assert_equal(
        quantile(x, probs=[0.0, 0.4, 0.5, 1.0]), torch.tensor([0.0, 0.8, 1.0, 2.0])
    )
    assert_equal(quantile(y, probs=0.2), torch.tensor(0.2), prec=0.02)
    assert_equal(quantile(z, probs=0.8413), torch.tensor(1.0), prec=0.02)


@pytest.mark.init(rng_seed=3)
def test_weighed_quantile():
    # Fixed values test
    input = torch.Tensor([[10, 50, 40], [20, 30, 0]])
    probs = [0.2, 0.8]
    log_weights = torch.Tensor([0.4, 0.5, 0.1]).log()
    result = weighed_quantile(input, probs, log_weights, -1)
    assert_equal(result, torch.Tensor([[40.4, 47.6], [9.0, 26.4]]))

    # Random values test
    dist = torch.distributions.normal.Normal(0, 1)
    input = dist.sample((100000,))
    probs = [0.1, 0.7, 0.95]
    log_weights = dist.log_prob(input)
    result = weighed_quantile(input, probs, log_weights)
    result_dist = torch.distributions.normal.Normal(0, torch.tensor(0.5).sqrt())
    assert_equal(result, result_dist.icdf(torch.Tensor(probs)), prec=0.01)


def test_pi():
    x = torch.randn(1000).exp()
    assert_equal(pi(x, prob=0.8), quantile(x, probs=[0.1, 0.9]))


@pytest.mark.init(rng_seed=3)
def test_hpdi():
    x = torch.randn(20000)
    assert_equal(hpdi(x, prob=0.8), pi(x, prob=0.8), prec=0.01)

    x = torch.empty(20000).exponential_(1)
    assert_equal(hpdi(x, prob=0.2), torch.tensor([0.0, 0.22]), prec=0.01)


def _quantile(x, dim=0):
    return quantile(x, probs=[0.1, 0.6], dim=dim)


def _pi(x, dim=0):
    return pi(x, prob=0.8, dim=dim)


def _hpdi(x, dim=0):
    return hpdi(x, prob=0.8, dim=dim)


@pytest.mark.parametrize("statistics", [_quantile, _pi, _hpdi])
@pytest.mark.parametrize("sample_shape", [(), (3,), (2, 3)])
def test_statistics_A_ok_with_sample_shape(statistics, sample_shape):
    xs = torch.rand((10,) + torch.Size(sample_shape))
    y = statistics(xs)

    # test correct shape
    assert_equal(y.shape, torch.Size([2]) + xs.shape[1:])

    # test correct batch calculation
    batch_statistics = []
    for x in xs.reshape(10, -1).split(1, dim=1):
        batch_statistics.append(statistics(x))
    assert_equal(torch.cat(batch_statistics, dim=1).reshape(y.shape), y)

    # test dim=-1
    a = xs.transpose(0, -1)
    assert_equal(statistics(a, dim=-1), y.transpose(0, -1))


def test_autocorrelation():
    x = torch.arange(10.0)
    actual = autocorrelation(x)
    assert_equal(
        actual,
        torch.tensor([1, 0.78, 0.52, 0.21, -0.13, -0.52, -0.94, -1.4, -1.91, -2.45]),
        prec=0.01,
    )


def test_autocorrelation_trivial():
    x = torch.zeros(10)
    actual = autocorrelation(x)
    assert_equal(actual, torch.ones(10), prec=0.01)


def test_autocorrelation_vectorized():
    # make a mostly noisy x with a couple constant series
    x = torch.randn(3, 4, 5)
    x[1, 2] = 0
    x[2, 3] = 1

    actual = autocorrelation(x, dim=-1)
    expected = torch.tensor([[autocorrelation(xij).tolist() for xij in xi] for xi in x])
    assert_equal(actual, expected)

    assert (actual[1, 2] == 1).all()
    assert (actual[2, 3] == 1).all()


def test_autocovariance():
    x = torch.arange(10.0)
    with xfail_if_not_implemented():
        actual = autocovariance(x)
    assert_equal(
        actual,
        torch.tensor(
            [8.25, 6.42, 4.25, 1.75, -1.08, -4.25, -7.75, -11.58, -15.75, -20.25]
        ),
        prec=0.01,
    )


def test_cummin():
    x = torch.rand(10)
    y = torch.empty(x.shape)
    y[0] = x[0]
    for i in range(1, x.size(0)):
        y[i] = min(x[i], y[i - 1])

    assert_equal(_cummin(x), y)


@pytest.mark.parametrize("statistics", [autocorrelation, autocovariance, _cummin])
@pytest.mark.parametrize("sample_shape", [(), (3,), (2, 3)])
def test_statistics_B_ok_with_sample_shape(statistics, sample_shape):
    xs = torch.rand((10,) + torch.Size(sample_shape))
    with xfail_if_not_implemented():
        y = statistics(xs)

    # test correct shape
    assert_equal(y.shape, xs.shape)

    # test correct batch calculation
    batch_statistics = []
    for x in xs.reshape(10, -1).split(1, dim=1):
        batch_statistics.append(statistics(x))
    assert_equal(torch.cat(batch_statistics, dim=1).reshape(xs.shape), y)

    # test dim=-1
    if statistics is not _cummin:
        a = xs.transpose(0, -1)
        assert_equal(statistics(a, dim=-1), y.transpose(0, -1))


def test_gelman_rubin():
    # only need to test precision for small data
    x = torch.empty(2, 10)
    x[0, :] = torch.arange(10.0)
    x[1, :] = torch.arange(10.0) + 1

    r_hat = gelman_rubin(x)
    assert_equal(r_hat.item(), 0.98, prec=0.01)


def test_split_gelman_rubin_agree_with_gelman_rubin():
    x = torch.rand(2, 10)
    r_hat1 = gelman_rubin(x.reshape(2, 2, 5).reshape(4, 5))
    r_hat2 = split_gelman_rubin(x)
    assert_equal(r_hat1, r_hat2)


def test_effective_sample_size():
    x = torch.arange(1000.0).reshape(100, 10)

    with xfail_if_not_implemented():
        # test against arviz
        assert_equal(effective_sample_size(x).item(), 52.64, prec=0.01)


@pytest.mark.parametrize(
    "diagnostics", [gelman_rubin, split_gelman_rubin, effective_sample_size]
)
@pytest.mark.parametrize("sample_shape", [(), (3,), (2, 3)])
def test_diagnostics_ok_with_sample_shape(diagnostics, sample_shape):
    sample_shape = torch.Size(sample_shape)
    xs = torch.rand((4, 100) + sample_shape)

    with xfail_if_not_implemented():
        y = diagnostics(xs)

        # test correct shape
        assert_equal(y.shape, sample_shape)

        # test correct batch calculation
        batch_diagnostics = []
        for x in xs.reshape(4, 100, -1).split(1, dim=2):
            batch_diagnostics.append(diagnostics(x))
        assert_equal(torch.cat(batch_diagnostics, dim=0).reshape(sample_shape), y)

        # test chain_dim, sample_dim at different positions
        a = xs.transpose(0, 1)
        b = xs.unsqueeze(-1).transpose(0, -1).squeeze(0)
        c = xs.unsqueeze(-1).transpose(1, -1).squeeze(1)
        assert_equal(diagnostics(a, chain_dim=1, sample_dim=0), y)
        assert_equal(diagnostics(b, chain_dim=-1, sample_dim=0), y)
        assert_equal(diagnostics(c, sample_dim=-1), y)


def test_waic():
    x = -torch.arange(1.0, 101).log().reshape(25, 4)
    w_pw, p_pw = waic(x, pointwise=True)
    w, p = waic(x)
    w1, p1 = waic(x.t(), dim=1)

    # test against loo package: http://mc-stan.org/loo/reference/waic.html
    assert_equal(w_pw, torch.tensor([7.49, 7.75, 7.86, 7.92]), prec=0.01)
    assert_equal(p_pw, torch.tensor([1.14, 0.91, 0.79, 0.70]), prec=0.01)

    assert_equal(w, w_pw.sum())
    assert_equal(p, p_pw.sum())

    assert_equal(w, w1)
    assert_equal(p, p1)


def test_weighted_waic():
    a = 1 + torch.rand(10)
    b = 1 + torch.rand(10)
    c = 1 + torch.rand(10)
    expanded_x = torch.stack([a, b, c, a, b, a, c, a, c]).log()
    x = torch.stack([a, b, c]).log()
    log_weights = torch.tensor([4.0, 2, 3]).log()
    # assume weights are unnormalized
    log_weights = log_weights - torch.randn(1)

    w1, p1 = waic(x, log_weights)
    w2, p2 = waic(expanded_x)

    # test lpd
    lpd1 = -0.5 * w1 + p1
    lpd2 = -0.5 * w2 + p2
    assert_equal(lpd1, lpd2)

    # test p_waic (also test for weighted_variance)
    unbiased_p1 = p1 * 2 / 3
    unbiased_p2 = p2 * 8 / 9
    assert_equal(unbiased_p1, unbiased_p2)

    # test correctness for dim=-1
    w3, p3 = waic(x.t(), log_weights, dim=-1)
    assert_equal(w1, w3)
    assert_equal(p1, p3)


@pytest.mark.parametrize("k", [0.2, 0.5])
@pytest.mark.parametrize("sigma", [0.8, 1.3])
def test_fit_generalized_pareto(k, sigma, n_samples=5000):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from scipy.stats import genpareto

    X = genpareto.rvs(c=k, scale=sigma, size=n_samples)
    fit_k, fit_sigma = fit_generalized_pareto(torch.tensor(X))
    assert_equal(k, fit_k, prec=0.02)
    assert_equal(sigma, fit_sigma, prec=0.02)


@pytest.mark.parametrize("event_shape", [(), (4,), (3, 2)])
@pytest.mark.parametrize("num_samples", [1, 2, 3, 4, 10])
def test_crps_univariate_energy_score_empirical(num_samples, event_shape):
    truth = torch.randn(event_shape)
    pred = truth + 0.1 * torch.randn((num_samples,) + event_shape)

    actual = crps_empirical(pred, truth)
    assert actual.shape == truth.shape

    expected = (pred - truth).abs().mean(0) - 0.5 * (
        pred - pred.unsqueeze(1)
    ).abs().mean([0, 1])
    assert_close(actual, expected)

    expected = energy_score_empirical(
        pred[..., None].swapaxes(0, -1)[0, ..., None], truth[..., None]
    )
    assert_close(actual, expected)


@pytest.mark.parametrize("sample_dim", [3, 10, 30, 100])
def test_multivariate_energy_score(sample_dim, num_samples=10000):
    pred_uncorrelated = torch.randn(num_samples, sample_dim)

    pred = torch.randn(num_samples, 1)
    pred = pred.expand(pred_uncorrelated.shape)

    truth = torch.randn(num_samples, 1)
    truth = truth.expand(pred_uncorrelated.shape)

    energy_score = energy_score_empirical(pred, truth).mean()
    energy_score_uncorrelated = energy_score_empirical(pred_uncorrelated, truth).mean()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from scipy.stats import chi

    assert_close(
        energy_score,
        torch.tensor(0.5 * chi(1).mean() * (2 * sample_dim) ** 0.5),
        rtol=0.02,
    )
    assert energy_score * 1.02 < energy_score_uncorrelated


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)])
@pytest.mark.parametrize("sample_dim", [30, 100])
@pytest.mark.parametrize(
    "num_samples, pred_batch_size", [(100, 10), (100, 30), (100, 100), (100, 200)]
)
def test_energy_score_empirical_batched_calculation(
    batch_shape, sample_dim, num_samples, pred_batch_size
):
    # Generate data
    truth = torch.randn(batch_shape + (sample_dim,))
    pred = torch.randn(batch_shape + (num_samples, sample_dim))
    # Do batched and regular calculation
    expected = energy_score_empirical(pred, truth)
    actual = energy_score_empirical(pred, truth, pred_batch_size=pred_batch_size)
    # Check accuracy
    assert_close(actual, expected)


def test_jit_compilation():
    # Test that functions can be JIT compiled
    torch.jit.script(energy_score_empirical)
