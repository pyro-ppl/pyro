from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import assert_equal


class TestCategorical(TestCase):
    """
    Tests methods specific to the Categorical distribution
    """

    def setUp(self):
        n = 1
        self.ps = Variable(torch.Tensor([0.1, 0.6, 0.3]))
        self.batch_ps = Variable(torch.Tensor([[0.1, 0.6, 0.3], [0.2, 0.4, 0.4]]))
        self.n = Variable(torch.Tensor([n]))
        self.test_data = Variable(torch.Tensor([2]))
        self.analytic_mean = n * self.ps
        one = Variable(torch.ones(3))
        self.analytic_var = n * torch.mul(self.ps, one.sub(self.ps))

        # Discrete Distribution
        self.d_ps = Variable(torch.Tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]))
        self.d_test_data = Variable(torch.Tensor([[0], [5]]))

        self.n_samples = 50000

        self.support_non_vec = torch.Tensor([0, 1, 2])
        self.support = torch.Tensor([[0, 0], [1, 1], [2, 2]])

    def test_log_pdf(self):
        log_px_torch = dist.categorical.log_prob(self.test_data, self.ps).sum().data[0]
        log_px_np = float(sp.multinomial.logpmf(np.array([0, 0, 1]), 1, self.ps.data.cpu().numpy()))
        assert_equal(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [dist.categorical(self.ps).data.cpu().numpy()
                         for _ in range(self.n_samples)]
        _, counts = np.unique(torch_samples, return_counts=True)
        computed_mean = float(counts[0]) / self.n_samples
        assert_equal(computed_mean, self.analytic_mean.data.cpu().numpy()[0], prec=0.05)

    def test_support_non_vectorized(self):
        s = dist.categorical.enumerate_support(self.d_ps[0].squeeze(0))
        assert_equal(s.data, self.support_non_vec)

    def test_support(self):
        s = dist.categorical.enumerate_support(self.d_ps)
        assert_equal(s.data, self.support)


def wrap_nested(x, dim):
    if dim == 0:
        return x
    return wrap_nested([x], dim-1)


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
    support = dist.categorical.enumerate_support(ps)
    assert_equal(support.size(), torch.Size((ps.size(-1),) + ps.size()[:-1]))


def test_sample_dims(dim, ps):
    ps = modify_params_using_dims(ps, dim)
    sample = dist.categorical.sample(ps)
    expected_shape = dist.categorical.shape(ps)
    if not expected_shape:
        expected_shape = torch.Size((1,))
    assert_equal(sample.size(), expected_shape)


def test_batch_log_dims(dim, ps):
    ps = modify_params_using_dims(ps, dim)
    log_prob_shape = torch.Size((3,) + dist.categorical.batch_shape(ps))
    support = dist.categorical.enumerate_support(ps)
    log_prob = dist.categorical.log_prob(support, ps)
    assert_equal(log_prob.size(), log_prob_shape)
