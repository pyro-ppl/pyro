# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import pytest
from pyro.contrib.oed.eig import EwmaLog
from tests.common import assert_equal


@pytest.mark.parametrize("alpha", [0.5, 0.9, 0.99])
def test_ewma(alpha, NS=10000, D=1):
    ewma_log = EwmaLog(alpha=alpha)
    sigma = torch.tensor(1.0, requires_grad=True)

    for k in range(1000):
        exponent = torch.randn(NS, D) * sigma
        s, _ = torch.max(exponent, dim=0)
        log_eT = s + ewma_log((exponent - s).exp().mean(dim=0), s)
        log_eT.backward()
        sigma_grad = sigma.grad.clone().cpu().numpy()
        sigma.grad.zero_()
        if k % 100 == 0:
            error = math.fabs(sigma_grad - 1.0)
            assert error < 0.07


def test_ewma_log():
    ewma_log = EwmaLog(alpha=0.5)
    input1 = torch.tensor(2.)
    ewma_log(input1, torch.tensor(0.))
    assert_equal(ewma_log.ewma, input1)
    input2 = torch.tensor(3.)
    ewma_log(input2, torch.tensor(0.))
    assert_equal(ewma_log.ewma, torch.tensor(8./3))


def test_ewma_log_with_s():
    ewma_log = EwmaLog(alpha=0.5)
    input1 = torch.tensor(-1.)
    s1 = torch.tensor(210.)
    ewma_log(input1, s1)
    assert_equal(ewma_log.ewma, input1)
    input2 = torch.tensor(-1.)
    s2 = torch.tensor(210.5)
    ewma_log(input2, s2)
    true_ewma = (1./3)*(torch.exp(s1 - s2)*input1 + 2*input2)
    assert_equal(ewma_log.ewma, true_ewma)
