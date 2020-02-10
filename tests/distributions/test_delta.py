# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase

import numpy as np
import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_equal


class TestDelta(TestCase):
    def setUp(self):
        self.v = torch.tensor([3.0])
        self.vs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.vs_expanded = self.vs.expand(4, 3)
        self.test_data = torch.tensor([[3.0], [3.0], [3.0]])
        self.batch_test_data_1 = torch.arange(0., 4.).unsqueeze(1).expand(4, 3)
        self.batch_test_data_2 = torch.arange(4., 8.).unsqueeze(1).expand(4, 3)
        self.batch_test_data_3 = torch.Tensor([[3.], [3.], [3.], [3.]])
        self.expected_support = [[[0.], [1.], [2.], [3.]]]
        self.expected_support_non_vec = [[3.]]
        self.analytic_mean = 3.
        self.analytic_var = 0.
        self.n_samples = 10

    def test_log_prob_sum(self):
        log_px_torch = dist.Delta(self.v).log_prob(self.test_data).sum()
        assert_equal(log_px_torch.item(), 0)

    def test_batch_log_prob(self):
        log_px_torch = dist.Delta(self.vs_expanded).log_prob(self.batch_test_data_1).data
        assert_equal(log_px_torch.sum().item(), 0)
        log_px_torch = dist.Delta(self.vs_expanded).log_prob(self.batch_test_data_2).data
        assert_equal(log_px_torch.sum().item(), float('-inf'))

    def test_batch_log_prob_shape(self):
        assert dist.Delta(self.vs).log_prob(self.batch_test_data_3).size() == (4, 1)
        assert dist.Delta(self.v).log_prob(self.batch_test_data_3).size() == (4, 1)

    def test_mean_and_var(self):
        torch_samples = [dist.Delta(self.v).sample().detach().cpu().numpy()
                         for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        assert_equal(torch_mean, self.analytic_mean)
        assert_equal(torch_var, self.analytic_var)


@pytest.mark.parametrize('batch_dim,event_dim',
                         [(b, e) for b in range(4) for e in range(1+b)])
@pytest.mark.parametrize('has_log_density', [False, True])
def test_shapes(batch_dim, event_dim, has_log_density):
    shape = tuple(range(2, 2 + batch_dim + event_dim))
    batch_shape = shape[:batch_dim]
    v = torch.randn(shape)
    log_density = torch.randn(batch_shape) if has_log_density else 0

    d = dist.Delta(v, log_density=log_density, event_dim=event_dim)
    x = d.rsample()
    assert (x == v).all()
    assert (d.log_prob(x) == log_density).all()


@pytest.mark.parametrize('batch_shape', [(), [], (2,), [2], torch.Size([2]), [2, 3]])
def test_expand(batch_shape):
    d1 = dist.Delta(torch.tensor(1.234))
    d2 = d1.expand(batch_shape)
    assert d2.batch_shape == torch.Size(batch_shape)
