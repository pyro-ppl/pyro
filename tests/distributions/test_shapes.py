import pytest

import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones


def test_diag_normal_shape():
    mu = ng_zeros(3, 2)
    sigma = ng_ones(3, 2)
    d = dist.DiagNormal(mu, sigma)
    assert d.batch_shape == (3,)
    assert d.event_shape == (2,)
    assert d.shape == (3, 2)
    assert d.sample().shape == d.shape


@pytest.mark.parametrize('one_hot', [True, False])
def test_categorical_shape(one_hot):
    ps = ng_ones(3, 2) / 2
    d = dist.Categorical(ps, one_hot=one_hot)
    assert d.batch_shape == (3,)
    if one_hot:
        assert d.event_shape == (2,)
        assert d.shape == (3, 2)
    else:
        assert d.event_shape == (1,)
        assert d.shape == (3, 1)
    assert d.sample().shape == d.shape


def test_dirichlet_shape():
    alpha = ng_ones(3, 2) / 2
    d = dist.Dirichlet(alpha)
    assert d.batch_shape == (3,)
    assert d.event_shape == (2,)
    assert d.shape == (3, 2)
    assert d.sample().shape == d.shape


def test_diag_normal_batch_log_pdf_shape():
    mu = ng_zeros(3, 2)
    sigma = ng_ones(3, 2)
    x = ng_zeros(3, 2)
    assert dist.DiagNormal(mu, sigma).batch_log_pdf(x).size() == (3,)


def test_bernoulli_batch_log_pdf_shape():
    ps = ng_ones(3, 2)
    x = ng_ones(3, 2)
    dist.Bernoulli(ps).batch_log_pdf(x).size() == (3,)


@pytest.mark.xfail
def test_categorical_batch_log_pdf_shape():
    ps = ng_ones(3, 2, 4) / 4
    x = ng_ones(3, 2)
    dist.Categorical(ps, one_hot=False).batch_log_pdf(x).size() == (3,)
