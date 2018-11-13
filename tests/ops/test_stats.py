import pytest
import torch

from pyro.ops.stats import (autocorrelation, autocovariance, effective_sample_size,
                            gelman_rubin, split_gelman_rubin,
                            _cummin, _fft_next_good_size)
from tests.common import assert_equal


def test_gelman_rubin():
    # only need to test precision for small data
    x = torch.empty(2, 10)
    x[0] = torch.arange(10.)
    x[1] = torch.arange(10.) + 1

    r_hat = gelman_rubin(x, sample_dim=1, chain_dim=0)
    assert_equal(r_hat.item(), 0.98, prec=0.01)

    # test shape
    y = torch.rand(3, 7, 10)
    assert_equal(gelman_rubin(y).shape, torch.Size([10]))
    assert_equal(gelman_rubin(y, sample_dim=-1, chain_dim=0).shape, torch.Size([7]))
    assert_equal(gelman_rubin(y, sample_dim=1, chain_dim=2).shape, torch.Size([3]))


def test_split_gelman_rubin():
    x = torch.rand(4, 10, 5)
    r_hat1 = gelman_rubin(x.reshape(4, 2, 5, 5).reshape(8, 5, 5), sample_dim=1, chain_dim=0)
    r_hat2 = split_gelman_rubin(x, sample_dim=1, chain_dim=0)
    assert_equal(r_hat1, r_hat2)


def test_autocorrelation():
    x = torch.arange(10.)
    assert_equal(autocorrelation(x),
                 torch.tensor([1, 0.78, 0.52, 0.21, -0.13,
                               -0.52, -0.94, -1.4, -1.91, -2.45]), prec=0.01)

    # test shape
    y = torch.rand(3, 7, 10)
    for i in range(3):
        assert_equal(autocorrelation(y, dim=i).shape, torch.Size([3, 7, 10]))


def test_autocovariance():
    x = torch.arange(10.)
    assert_equal(autocovariance(x),
                 torch.tensor([8.25, 6.42, 4.25, 1.75, -1.08,
                               -4.25, -7.75, -11.58, -15.75, -20.25]), prec=0.01)

    # test shape
    y = torch.rand(3, 7, 10)
    for i in range(3):
        assert_equal(autocovariance(y, dim=i).shape, torch.Size([3, 7, 10]))


def test_fft_next_good_size():
    assert_equal(_fft_next_good_size(433), 450)


def test_cummin():
    x = torch.rand(10)
    y = torch.empty(x.shape)
    y[0] = x[0]
    for i in range(1, x.size(0)):
        y[i] = min(x[i], y[i-1])

    assert_equal(_cummin(x), y)


@pytest.mark.init(rng_seed=3)
def test_effective_sample_size():
    x = torch.empty(2000)
    x[0].normal_(0, 1)
    for i in range(1, x.size(0)):
        x[i].normal_(0.8 * x[i-1], 1)
    x = x.reshape(4, x.size(0) // 4)

    try:
        assert_equal(effective_sample_size(x, sample_dim=1, chain_dim=0).item(), 268, prec=2)

        # test shape
        y = torch.rand(3, 2, 7)
        assert_equal(effective_sample_size(y).shape, torch.Size([7]))
        assert_equal(effective_sample_size(y, sample_dim=-1, chain_dim=0).shape, torch.Size([2]))
        assert_equal(effective_sample_size(y, sample_dim=1, chain_dim=2).shape, torch.Size([3]))
    except NotImplementedError:
        pytest.skip()
