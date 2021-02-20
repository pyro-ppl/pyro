# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.contrib.tracking.distributions import EKFDistribution
from pyro.contrib.tracking.dynamic_models import NcpContinuous, NcvContinuous

import pytest


@pytest.mark.parametrize('Model', [NcpContinuous, NcvContinuous])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('time', [2, 3])
def test_EKFDistribution_smoke(Model, dim, time):
    x0 = torch.rand(2*dim)
    ys = torch.randn(time, dim)
    P0 = torch.eye(2*dim).requires_grad_()
    R = torch.eye(dim).requires_grad_()
    model = Model(2*dim, 2.0)
    dist = EKFDistribution(x0, P0, model, R, time_steps=time)
    log_prob = dist.log_prob(ys)
    assert log_prob.shape == torch.Size()
    dP0, dR = torch.autograd.grad(log_prob, [P0, R])
    assert dP0.shape == P0.shape
    assert dR.shape == R.shape
