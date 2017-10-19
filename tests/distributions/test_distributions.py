import numpy as np
import pytest
import torch

from tests.common import assert_equal, xfail_if_not_implemented


def unwrap_variable(x):
    return x.data.cpu().numpy()


# Distribution tests - all distributions

def test_log_pdf(dist, test_data_idx):
    pyro_log_pdf = unwrap_variable(dist.get_pyro_logpdf(test_data_idx))[0]
    scipy_log_pdf = dist.get_scipy_logpdf(test_data_idx)
    assert_equal(pyro_log_pdf, scipy_log_pdf)


def test_batch_log_pdf(dist):
    # TODO (@npradhan) - remove once #144 is resolved
    try:
        logpdf_sum_pyro = unwrap_variable(torch.sum(dist.get_pyro_batch_logpdf()))[0]
    except NotImplementedError:
        pytest.skip("Batch log pdf not tested for distribution")
    logpdf_sum_np = np.sum(dist.get_scipy_batch_logpdf())
    assert_equal(logpdf_sum_pyro, logpdf_sum_np)


def test_shape(dist):
    d = dist.pyro_dist
    args = dist.get_dist_params(0)
    with xfail_if_not_implemented():
        assert d.shape(*args) == d.batch_shape(*args) + d.event_shape(*args)


def test_sample_shape(dist):
    d = dist.pyro_dist
    args = dist.get_dist_params(0)
    x = dist.pyro_dist.sample(*args)
    with xfail_if_not_implemented():
        assert x.size() == d.shape(*args)


def test_batch_log_pdf_shape(dist):
    if dist.pyro_dist.__class__.__name__ == 'Multinomial':
        pytest.xfail('Fixture parameters are not tensors')
    d = dist.pyro_dist
    args = dist.get_dist_params(0)
    x = d.sample(*args)
    with xfail_if_not_implemented():
        log_p = d.batch_log_pdf(x, *args)
        assert log_p.size() == d.batch_shape(*args) + (1,)


def test_mean_and_variance(dist, test_data_idx):
    num_samples = dist.get_num_samples(test_data_idx)
    dist_params = dist.get_dist_params(test_data_idx)
    torch_samples = dist.get_samples(num_samples, *dist_params)
    sample_mean = np.mean(torch_samples, 0)
    sample_var = np.var(torch_samples, 0)
    try:
        analytic_mean = unwrap_variable(dist.pyro_dist.analytic_mean(*dist_params))
        analytic_var = unwrap_variable(dist.pyro_dist.analytic_var(*dist_params))
        assert_equal(sample_mean, analytic_mean, prec=dist.prec)
        assert_equal(sample_var, analytic_var, prec=dist.prec)
    except (NotImplementedError, ValueError):
        pytest.skip('analytic mean and variance are not available')


# Distributions tests - discrete distributions

def test_support(discrete_dist):
    expected_support = discrete_dist.expected_support
    expected_support_non_vec = discrete_dist.expected_support_non_vec
    if not expected_support:
        pytest.skip("Support not tested for distribution")
    actual_support_non_vec = discrete_dist.pyro_dist.support(*discrete_dist.get_dist_params(0))
    actual_support = discrete_dist.pyro_dist.support(*discrete_dist.get_dist_params())
    assert_equal(actual_support.data, torch.Tensor(expected_support))
    assert_equal(actual_support_non_vec.data, torch.Tensor(expected_support_non_vec))
