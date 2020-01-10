# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase

import numpy as np
import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_equal


class TestOneHotCategorical(TestCase):
    """
    Tests methods specific to the OneHotCategorical distribution
    """

    def setUp(self):
        n = 1
        self.probs = torch.tensor([0.1, 0.6, 0.3])
        self.batch_ps = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]])
        self.n = torch.tensor([n])
        self.test_data = torch.tensor([0.0, 1.0, 0.0])
        self.test_data_nhot = torch.tensor([2.0])
        self.analytic_mean = n * self.probs
        one = torch.ones(3)
        self.analytic_var = n * torch.mul(self.probs, one.sub(self.probs))

        # Discrete Distribution
        self.d_ps = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
        self.d_test_data = torch.tensor([[0.0], [5.0]])
        self.d_v_test_data = [['a'], ['f']]

        self.n_samples = 50000

        self.support_one_hot_non_vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.support_one_hot = torch.tensor([[[1, 0, 0], [1, 0, 0]],
                                             [[0, 1, 0], [0, 1, 0]],
                                             [[0, 0, 1], [0, 0, 1]]])
        self.support_non_vec = torch.LongTensor([[0], [1], [2]])
        self.support = torch.LongTensor([[[0], [0]], [[1], [1]], [[2], [2]]])
        self.discrete_support_non_vec = torch.tensor([[0.0], [1.0], [2.0]])
        self.discrete_support = torch.tensor([[[0.0], [3.0]], [[1.0], [4.0]], [[2.0], [5.0]]])
        self.discrete_arr_support_non_vec = [['a'], ['b'], ['c']]
        self.discrete_arr_support = [[['a'], ['d']], [['b'], ['e']], [['c'], ['f']]]

    def test_support_non_vectorized(self):
        s = dist.OneHotCategorical(self.d_ps[0].squeeze(0)).enumerate_support()
        assert_equal(s.data, self.support_one_hot_non_vec)

    def test_support(self):
        s = dist.OneHotCategorical(self.d_ps).enumerate_support()
        assert_equal(s.data, self.support_one_hot)


def wrap_nested(x, dim):
    if dim == 0:
        return x
    return wrap_nested([x], dim-1)


def assert_correct_dimensions(sample, probs):
    ps_shape = list(probs.data.size())
    sample_shape = list(sample.shape)
    assert_equal(sample_shape, ps_shape)


@pytest.fixture(params=[1, 2, 3], ids=lambda x: "dim=" + str(x))
def dim(request):
    return request.param


@pytest.fixture(params=[[0.3, 0.5, 0.2]], ids=None)
def probs(request):
    return request.param


def modify_params_using_dims(probs, dim):
    return torch.tensor(wrap_nested(probs, dim-1))


def test_support_dims(dim, probs):
    probs = modify_params_using_dims(probs, dim)
    d = dist.OneHotCategorical(probs)
    support = d.enumerate_support()
    for s in support:
        assert_correct_dimensions(s, probs)
    n = len(support)
    assert support.shape == (n,) + d.batch_shape + d.event_shape
    support_expanded = d.enumerate_support(expand=True)
    assert support_expanded.shape == (n,) + d.batch_shape + d.event_shape
    support_unexpanded = d.enumerate_support(expand=False)
    assert support_unexpanded.shape == (n,) + (1,) * len(d.batch_shape) + d.event_shape


def test_sample_dims(dim, probs):
    probs = modify_params_using_dims(probs, dim)
    sample = dist.OneHotCategorical(probs).sample()
    assert_correct_dimensions(sample, probs)


def test_batch_log_dims(dim, probs):
    batch_pdf_shape = (3,) + (1,) * (dim-1)
    expected_log_prob_sum = np.array(wrap_nested(list(np.log(probs)), dim-1)).reshape(*batch_pdf_shape)
    probs = modify_params_using_dims(probs, dim)
    support = dist.OneHotCategorical(probs).enumerate_support()
    log_prob = dist.OneHotCategorical(probs).log_prob(support)
    assert_equal(log_prob.detach().cpu().numpy(), expected_log_prob_sum)
