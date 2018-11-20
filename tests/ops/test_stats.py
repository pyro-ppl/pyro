import pytest
import torch

from pyro.ops.stats import (autocorrelation, autocovariance, effective_sample_size,
                            gelman_rubin, split_gelman_rubin,
                            _cummin, _fft_next_good_size)
from tests.common import assert_equal, xfail_if_not_implemented


def test_autocorrelation():
    x = torch.arange(10.)
    with xfail_if_not_implemented():
        actual = autocorrelation(x)
    assert_equal(actual,
                 torch.tensor([1, 0.78, 0.52, 0.21, -0.13,
                               -0.52, -0.94, -1.4, -1.91, -2.45]), prec=0.01)


def test_autocovariance():
    x = torch.arange(10.)
    with xfail_if_not_implemented():
        actual = autocovariance(x)
    assert_equal(actual,
                 torch.tensor([8.25, 6.42, 4.25, 1.75, -1.08,
                               -4.25, -7.75, -11.58, -15.75, -20.25]), prec=0.01)


def test_cummin():
    x = torch.rand(10)
    y = torch.empty(x.shape)
    y[0] = x[0]
    for i in range(1, x.size(0)):
        y[i] = min(x[i], y[i-1])

    assert_equal(_cummin(x), y)


@pytest.mark.parametrize('statistics', [autocorrelation, autocovariance, _cummin])
@pytest.mark.parametrize('sample_shape', [(), (3,), (2, 3)])
def test_statistics_ok_with_sample_shape(statistics, sample_shape):
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


def test_fft_next_good_size():
    assert_equal(_fft_next_good_size(433), 450)


def test_gelman_rubin():
    # only need to test precision for small data
    x = torch.empty(10, 2)
    x[:, 0] = torch.arange(10.)
    x[:, 1] = torch.arange(10.) + 1

    r_hat = gelman_rubin(x)
    assert_equal(r_hat.item(), 0.98, prec=0.01)


def test_split_gelman_rubin_agree_with_gelman_rubin():
    x = torch.rand(10, 2)
    r_hat1 = gelman_rubin(x.reshape(2, 5, 2).transpose(0, 1).reshape(5, 4))
    r_hat2 = split_gelman_rubin(x)
    assert_equal(r_hat1, r_hat2)


@pytest.mark.init(rng_seed=3)
def test_effective_sample_size():
    x = torch.empty(1000, 2)
    x[0, 0].normal_(0, 1)
    x[0, 1].normal_(0, 1)
    for i in range(1, x.size(0)):
        x[i, 0].normal_(0.8 * x[i - 1, 0], 1)
        x[i, 1].normal_(0.9 * x[i - 1, 1], 1)

    with xfail_if_not_implemented():
        assert_equal(effective_sample_size(x).item(), 134.5, prec=1)


@pytest.mark.parametrize('diagnostics', [gelman_rubin, split_gelman_rubin, effective_sample_size])
@pytest.mark.parametrize('sample_shape', [(), (3,), (2, 3)])
def test_diagnostics_ok_with_sample_shape(diagnostics, sample_shape):
    sample_shape = torch.Size(sample_shape)
    xs = torch.rand((100, 4) + sample_shape)

    with xfail_if_not_implemented():
        y = diagnostics(xs)

        # test correct shape
        assert_equal(y.shape, sample_shape)

        # test correct batch calculation
        batch_diagnostics = []
        for x in xs.reshape(100, 4, -1).split(1, dim=2):
            batch_diagnostics.append(diagnostics(x))
        assert_equal(torch.cat(batch_diagnostics, dim=0).reshape(sample_shape), y)

        # test chain_dim, sample_dim at different positions
        a = xs.transpose(0, 1)
        b = xs.unsqueeze(-1).transpose(0, -1).squeeze(0)
        c = xs.unsqueeze(-1).transpose(1, -1).squeeze(1)
        assert_equal(diagnostics(a, sample_dim=1, chain_dim=0), y)
        assert_equal(diagnostics(b, sample_dim=-1, chain_dim=0), y)
        assert_equal(diagnostics(c, chain_dim=-1), y)
