# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import scipy.stats as sp
import torch

import pyro.distributions as dist
from tests.common import assert_equal


@pytest.fixture()
def test_data():
    return torch.DoubleTensor([0.4])


@pytest.fixture()
def alpha():
    """
    alpha parameter for the Beta distribution.
    """
    return torch.DoubleTensor([2.4])


@pytest.fixture()
def beta():
    """
    beta parameter for the Beta distribution.
    """
    return torch.DoubleTensor([3.7])


@pytest.fixture()
def float_test_data(test_data):
    return torch.FloatTensor(test_data.detach().cpu().numpy())


@pytest.fixture()
def float_alpha(alpha):
    return torch.FloatTensor(alpha.detach().cpu().numpy())


@pytest.fixture()
def float_beta(beta):
    return torch.FloatTensor(beta.detach().cpu().numpy())


def test_double_type(test_data, alpha, beta):
    log_px_torch = dist.Beta(alpha, beta).log_prob(test_data).data
    assert isinstance(log_px_torch, torch.DoubleTensor)
    log_px_val = log_px_torch.numpy()
    log_px_np = sp.beta.logpdf(
        test_data.detach().cpu().numpy(),
        alpha.detach().cpu().numpy(),
        beta.detach().cpu().numpy())
    assert_equal(log_px_val, log_px_np, prec=1e-4)


def test_float_type(float_test_data, float_alpha, float_beta, test_data, alpha, beta):
    log_px_torch = dist.Beta(float_alpha, float_beta).log_prob(float_test_data).data
    assert isinstance(log_px_torch, torch.FloatTensor)
    log_px_val = log_px_torch.numpy()
    log_px_np = sp.beta.logpdf(
        test_data.detach().cpu().numpy(),
        alpha.detach().cpu().numpy(),
        beta.detach().cpu().numpy())
    assert_equal(log_px_val, log_px_np, prec=1e-4)


def test_conflicting_types(test_data, float_alpha, beta):
    with pytest.raises((TypeError, RuntimeError)):
        dist.Beta(float_alpha, beta).log_prob(test_data)
