from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np
import pytest
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import assert_equal


class TestOneHotCategorical(TestCase):
    """
    Tests methods specific to the OneHotCategorical distribution
    """

    def setUp(self):
        n = 1
        self.ps = Variable(torch.Tensor([0.1, 0.6, 0.3]))
        self.batch_ps = Variable(torch.Tensor([[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]]))
        self.n = Variable(torch.Tensor([n]))
        self.test_data = Variable(torch.Tensor([0, 1, 0]))
        self.test_data_nhot = Variable(torch.Tensor([2]))
        self.analytic_mean = n * self.ps
        one = Variable(torch.ones(3))
        self.analytic_var = n * torch.mul(self.ps, one.sub(self.ps))

        # Discrete Distribution
        self.d_ps = Variable(torch.Tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]))
        self.d_test_data = Variable(torch.Tensor([[0], [5]]))
        self.d_v_test_data = [['a'], ['f']]

        self.n_samples = 50000

        self.support_one_hot_non_vec = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.support_one_hot = torch.Tensor([[[1, 0, 0], [1, 0, 0]],
                                             [[0, 1, 0], [0, 1, 0]],
                                             [[0, 0, 1], [0, 0, 1]]])
        self.support_non_vec = torch.LongTensor([[0], [1], [2]])
        self.support = torch.LongTensor([[[0], [0]], [[1], [1]], [[2], [2]]])
        self.discrete_support_non_vec = torch.Tensor([[0], [1], [2]])
        self.discrete_support = torch.Tensor([[[0], [3]], [[1], [4]], [[2], [5]]])
        self.discrete_arr_support_non_vec = [['a'], ['b'], ['c']]
        self.discrete_arr_support = [[['a'], ['d']], [['b'], ['e']], [['c'], ['f']]]

    def test_support_non_vectorized(self):
        s = dist.one_hot_categorical.enumerate_support(self.d_ps[0].squeeze(0))
        assert_equal(s.data, self.support_one_hot_non_vec)

    def test_support(self):
        s = dist.one_hot_categorical.enumerate_support(self.d_ps)
        assert_equal(s.data, self.support_one_hot)


def wrap_nested(x, dim):
    if dim == 0:
        return x
    return wrap_nested([x], dim-1)


def assert_correct_dimensions(sample, ps):
    ps_shape = list(ps.data.size())
    if isinstance(sample, torch.autograd.Variable):
        sample_shape = list(sample.data.size())
    else:
        sample_shape = list(sample.shape)
    assert_equal(sample_shape, ps_shape)


@pytest.fixture(params=[1, 2, 3], ids=lambda x: "dim=" + str(x))
def dim(request):
    return request.param


@pytest.fixture(params=[[0.3, 0.5, 0.2]], ids=None)
def ps(request):
    return request.param


def modify_params_using_dims(ps, dim):
    return Variable(torch.Tensor(wrap_nested(ps, dim-1)))


def test_support_dims(dim, ps):
    ps = modify_params_using_dims(ps, dim)
    support = dist.one_hot_categorical.enumerate_support(ps)
    for s in support:
        assert_correct_dimensions(s, ps)


def test_sample_dims(dim, ps):
    ps = modify_params_using_dims(ps, dim)
    sample = dist.one_hot_categorical.sample(ps)
    assert_correct_dimensions(sample, ps)


def test_batch_log_dims(dim, ps):
    batch_pdf_shape = (3,) + (1,) * dim
    expected_log_pdf = np.array(wrap_nested(list(np.log(ps)), dim-1)).reshape(*batch_pdf_shape)
    ps = modify_params_using_dims(ps, dim)
    support = dist.one_hot_categorical.enumerate_support(ps)
    batch_log_pdf = dist.one_hot_categorical.batch_log_pdf(support, ps)
    assert_equal(batch_log_pdf.data.cpu().numpy(), expected_log_pdf)
