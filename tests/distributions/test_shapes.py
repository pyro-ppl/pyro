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


def test_diag_normal_batch_log_pdf_shape():
    mu1 = ng_zeros(2, 3)
    mu2 = ng_zeros(2, 4)
    sigma = ng_zeros(2, 1)
    # TODO these will need extra_event_dims=1
    d1 = dist.Normal(mu1, sigma.expand_as(mu1))
    d2 = dist.Normal(mu2, sigma.expand_as(mu2))
    x1 = d1.sample()
    x2 = d2.sample()
    assert d1.batch_log_pdf(x1).size() == (2, 1)  # TODO will become (2,)
    assert d2.batch_log_pdf(x2).size() == (2, 1)  # TODO will become (2,)
    d1.batch_log_pdf(x1) + d2.batch_log_pdf(x2)  # this line should not error


def test_diag_normal_log_pdf_mask_shape():
    mu = ng_zeros(4, 3)
    sigma = ng_zeros(4, 3)
    mask = ng_ones(4, 1)  # note that 3 was summed out  TODO will become (4,)
    # TODO this will need extra_event_dims=1
    d = dist.Normal(mu, sigma, log_pdf_mask=mask)
    x = d.sample()
    assert d.batch_log_pdf(x).size() == (4, 1)  # TODO will become (4,)
