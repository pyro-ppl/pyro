from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
from pyro.util import ng_ones, ng_zeros


def test_categorical_shape():
    ps = ng_ones(3, 2) / 2
    d = dist.Categorical(ps)
    assert d.batch_shape() == (3,)
    assert d.event_shape() == (1,)
    assert d.shape() == (3, 1)
    assert d.sample().size() == d.shape()


def test_one_hot_categorical_shape():
    ps = ng_ones(3, 2) / 2
    d = dist.OneHotCategorical(ps)
    assert d.batch_shape() == (3,)
    assert d.event_shape() == (2,)
    assert d.shape() == (3, 2)
    assert d.sample().size() == d.shape()


def test_normal_shape():
    mu = ng_zeros(3, 2)
    sigma = ng_ones(3, 2)
    d = dist.Normal(mu, sigma)
    assert d.batch_shape() == (3,)
    assert d.event_shape() == (2,)
    assert d.shape() == (3, 2)
    assert d.sample().size() == d.shape()


def test_dirichlet_shape():
    alpha = ng_ones(3, 2) / 2
    d = dist.Dirichlet(alpha)
    assert d.batch_shape() == (3,)
    assert d.event_shape() == (2,)
    assert d.shape() == (3, 2)
    assert d.sample().size() == d.shape()


def test_bernoulli_batch_log_pdf_shape():
    ps = ng_ones(3, 2)
    x = ng_ones(3, 2)
    d = dist.Bernoulli(ps)
    assert d.batch_log_pdf(x).size() == (3, 1)


def test_categorical_batch_log_pdf_shape():
    ps = ng_ones(3, 2, 4) / 4
    x = ng_zeros(3, 2, 1)
    d = dist.Categorical(ps)
    assert d.batch_log_pdf(x).size() == (3, 2, 1)


def test_one_hot_categorical_batch_log_pdf_shape():
    ps = ng_ones(3, 2, 4) / 4
    x = ng_zeros(3, 2, 4)
    x[:, :, 0] = 1
    d = dist.OneHotCategorical(ps)
    assert d.batch_log_pdf(x).size() == (3, 2, 1)


def test_normal_batch_log_pdf_shape():
    mu = ng_zeros(3, 2)
    sigma = ng_ones(3, 2)
    x = ng_zeros(3, 2)
    d = dist.Normal(mu, sigma)
    assert d.batch_log_pdf(x).size() == (3, 1)
