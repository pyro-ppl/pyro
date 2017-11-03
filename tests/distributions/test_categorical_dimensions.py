from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch
from torch.autograd import Variable

import pyro.distributions as dist
from tests.common import assert_equal


def wrap_nested(x, dim):
    if dim == 0:
        return x
    return wrap_nested([x], dim-1)


def assert_correct_dimensions(sample, ps, vs, one_hot):
    ps_shape = list(ps.data.size())
    if isinstance(sample, torch.autograd.Variable):
        sample_shape = list(sample.data.size())
    else:
        sample_shape = list(sample.shape)
    if one_hot and not vs:
        assert_equal(sample_shape, ps_shape)
    else:
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


@pytest.fixture(params=[True, False], ids=lambda x: "one_hot=" + str(x))
def one_hot(request):
    return request.param


def modify_params_using_dims(ps, vs, dim):
    ps = Variable(torch.Tensor(wrap_nested(ps, dim-1)))
    if vs:
        vs = wrap_nested(vs, dim-1)
    return ps, vs


def test_support_dims(dim, vs, one_hot, ps):
    ps, vs = modify_params_using_dims(ps, vs, dim)
    support = dist.categorical.enumerate_support(ps, vs, one_hot=one_hot)
    for s in support:
        assert_correct_dimensions(s, ps, vs, one_hot)


def test_sample_dims(dim, vs, one_hot, ps):
    ps, vs = modify_params_using_dims(ps, vs, dim)
    sample = dist.categorical.sample(ps, vs, one_hot=one_hot)
    assert_correct_dimensions(sample, ps, vs, one_hot)


def test_batch_log_dims(dim, vs, one_hot, ps):
    batch_pdf_shape = (3,) + (1,) * dim
    expected_log_pdf = np.array(wrap_nested(list(np.log(ps)), dim-1)).reshape(*batch_pdf_shape)
    ps, vs = modify_params_using_dims(ps, vs, dim)
    support = dist.categorical.enumerate_support(ps, vs, one_hot=one_hot)
    batch_log_pdf = dist.categorical.batch_log_pdf(support, ps, vs, one_hot=one_hot)
    assert_equal(batch_log_pdf.data.cpu().numpy(), expected_log_pdf)
