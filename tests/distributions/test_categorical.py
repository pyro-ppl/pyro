from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import scipy.stats as sp
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import assert_equal


class TestCategorical(object):
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
        self.d_vs = Variable(torch.Tensor([[0, 1, 2], [3, 4, 5]]))
        self.d_vs_arr = [['a', 'b', 'c'], ['d', 'e', 'f']]
        self.d_vs_tup = (('a', 'b', 'c'), ('d', 'e', 'f'))
        self.d_test_data = Variable(torch.Tensor([[0], [5]]))
        self.d_v_test_data = [['a'], ['f']]

        self.n_samples = 50000

        self.support_non_vec = torch.Tensor([[0], [1], [2]])
        self.support = torch.Tensor([[[0], [3]], [[1], [4]], [[2], [5]]])
        self.arr_support_non_vec = [['a'], ['b'], ['c']]
        self.arr_support = [[['a'], ['d']], [['b'], ['e']], [['c'], ['f']]]

    def test_log_pdf(self):
        log_px_torch = dist.categorical.batch_log_pdf(self.test_data, self.ps).data[0]
        log_px_np = float(sp.multinomial.logpmf(np.array([0, 0, 1]), 1, self.ps.data.cpu().numpy()))
        assert_equal(log_px_torch, log_px_np, prec=1e-4)

    def test_mean_and_var(self):
        torch_samples = [dist.categorical(self.ps).data.cpu().numpy()
                         for _ in range(self.n_samples)]
        _, counts = np.unique(torch_samples, return_counts=True)
        computed_mean = float(counts[0]) / self.n_samples
        assert_equal(computed_mean, self.analytic_mean.data.cpu().numpy()[0], prec=0.05)

    def test_support_non_vectorized(self):
        s = dist.categorical.enumerate_support(self.d_ps[0].squeeze(0), self.d_vs[0].squeeze(0))
        assert_equal(s.data, self.support_non_vec)

    def test_support(self):
        s = dist.categorical.enumerate_support(self.d_ps, self.d_vs)
        assert_equal(s.data, self.support)

    def test_arr_support_non_vectorized(self):
        s = dist.categorical.enumerate_support(self.d_ps[0].squeeze(0), self.d_vs_arr[0]).tolist()
        assert_equal(s, self.arr_support_non_vec)

    def test_arr_support(self):
        s = dist.categorical.enumerate_support(self.d_ps, self.d_vs_arr).tolist()
        assert_equal(s, self.arr_support)


def wrap_nested(x, dim):
    if dim == 0:
        return x
    return wrap_nested([x], dim-1)


def assert_correct_dimensions(sample, ps, vs):
    ps_shape = list(ps.data.size())
    if isinstance(sample, torch.autograd.Variable):
        sample_shape = list(sample.data.size())
    else:
        sample_shape = list(sample.shape)
    assert_equal(sample_shape, ps_shape[:-1] + [1])


@pytest.fixture(params=[1, 2, 3], ids=lambda x: "dim=" + str(x))
def dim(request):
    return request.param


@pytest.fixture(params=[[0.3, 0.5, 0.2]], ids=None)
def ps(request):
    return request.param


@pytest.fixture(params=[None, [3, 4, 5], ["a", "b", "c"]],
                ids=["vs=None", "vs=list(num)", "vs=list(str)"])
def vs(request):
    return request.param


def modify_params_using_dims(ps, vs, dim):
    ps = Variable(torch.Tensor(wrap_nested(ps, dim-1)))
    if vs:
        vs = wrap_nested(vs, dim-1)
    return ps, vs


def test_support_dims(dim, vs, ps):
    ps, vs = modify_params_using_dims(ps, vs, dim)
    support = dist.categorical.enumerate_support(ps, vs)
    for s in support:
        assert_correct_dimensions(s, ps, vs)


def test_sample_dims(dim, vs, ps):
    ps, vs = modify_params_using_dims(ps, vs, dim)
    sample = dist.categorical.sample(ps, vs)
    assert_correct_dimensions(sample, ps, vs)


def test_batch_log_dims(dim, vs, ps):
    batch_pdf_shape = (3,) + (1,) * dim
    expected_log_pdf = np.array(wrap_nested(list(np.log(ps)), dim-1)).reshape(*batch_pdf_shape)
    ps, vs = modify_params_using_dims(ps, vs, dim)
    support = dist.categorical.enumerate_support(ps, vs)
    batch_log_pdf = dist.categorical.batch_log_pdf(support, ps, vs)
    assert_equal(batch_log_pdf.data.cpu().numpy(), expected_log_pdf)
