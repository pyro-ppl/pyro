import numpy as np
import pytest
import torch

from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal, xfail_if_not_implemented

SINGLE_TEST_DATUM_IDX = 0
BATCH_TEST_DATA_IDX = -1


def unwrap_variable(x):
    return x.data.cpu().numpy()


# Distribution tests - all distributions


def test_log_pdf(dist):
    pyro_log_pdf = unwrap_variable(dist.get_pyro_logpdf(SINGLE_TEST_DATUM_IDX))[0]
    scipy_log_pdf = dist.get_scipy_logpdf(SINGLE_TEST_DATUM_IDX)
    assert_equal(pyro_log_pdf, scipy_log_pdf)


def test_batch_log_pdf(dist):
    # TODO (@npradhan) - remove once #144 is resolved
    try:
        logpdf_sum_pyro = unwrap_variable(torch.sum(dist.get_pyro_batch_logpdf(BATCH_TEST_DATA_IDX)))[0]
    except NotImplementedError:
        pytest.skip("Batch log pdf not tested for distribution")
    logpdf_sum_np = np.sum(dist.get_scipy_batch_logpdf(-1))
    assert_equal(logpdf_sum_pyro, logpdf_sum_np)


def test_shape(dist):
    if dist.get_test_distribution_name() == 'Bernoulli':
        pytest.skip('FIXME: #323')
    d = dist.pyro_dist
    dist_params = dist.get_dist_params(SINGLE_TEST_DATUM_IDX)
    with xfail_if_not_implemented():
        assert d.shape(**dist_params) == d.batch_shape(**dist_params) + d.event_shape(**dist_params)


def test_sample_shape(dist):
    d = dist.pyro_dist
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x_func = dist.pyro_dist.sample(**dist_params)
        x_obj = dist.pyro_dist_obj(**dist_params).sample()
        assert_equal(x_obj.size(), x_func.size())
        with xfail_if_not_implemented():
            # TODO remove once #323 is resolved
            if dist.get_test_distribution_name() == 'Bernoulli':
                assert(x_func.size() == d.shape(x_func, **dist_params))
                return
            assert x_func.size() == d.shape(**dist_params)


def test_batch_log_pdf_shape(dist):
    if dist.pyro_dist.__class__.__name__ == 'Multinomial':
        pytest.xfail('FIXME: https://github.com/uber/pyro/issues/301')
    d = dist.pyro_dist
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x = dist.get_test_data(idx)
        with xfail_if_not_implemented():
            # Get batch pdf shape after broadcasting.
            expected_shape = get_batch_pdf_shape(dist, x, dist_params)
            log_p_func = d.batch_log_pdf(x, **dist_params)
            log_p_obj = dist.pyro_dist_obj(**dist_params).batch_log_pdf(x)
            # assert that the functional and object forms return
            # the same batch pdf.
            assert_equal(log_p_func.size(), log_p_obj.size())
            assert log_p_func.size() == expected_shape


def test_batch_log_pdf_mask(dist):
    if dist.pyro_dist.__class__.__name__ not in ('DiagNormal', 'Bernoulli', 'Categorical'):
        pytest.skip('Batch pdf masking not supported for the distribution.')
    d = dist.pyro_dist
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x = dist.get_test_data(idx)
        with xfail_if_not_implemented():
            batch_pdf_shape = get_batch_pdf_shape(dist, x, dist_params)
            zeros_mask = ng_zeros(batch_pdf_shape)
            ones_mask = ng_ones(batch_pdf_shape)
            half_mask = ng_ones(batch_pdf_shape) * 0.5
            batch_log_pdf = d.batch_log_pdf(x, **dist_params)
            batch_log_pdf_zeros_mask = d.batch_log_pdf(x, log_pdf_mask=zeros_mask, **dist_params)
            batch_log_pdf_ones_mask = d.batch_log_pdf(x, log_pdf_mask=ones_mask, **dist_params)
            batch_log_pdf_half_mask = d.batch_log_pdf(x, log_pdf_mask=half_mask, **dist_params)
            assert_equal(batch_log_pdf_ones_mask, batch_log_pdf)
            assert_equal(batch_log_pdf_zeros_mask, ng_zeros(batch_pdf_shape))
            assert_equal(batch_log_pdf_half_mask, 0.5 * batch_log_pdf)


def test_mean_and_variance(dist):
    num_samples = dist.get_num_samples(SINGLE_TEST_DATUM_IDX)
    dist_params = dist.get_dist_params(SINGLE_TEST_DATUM_IDX)
    torch_samples = dist.get_samples(num_samples, **dist_params)
    sample_mean = np.mean(torch_samples, 0)
    sample_var = np.var(torch_samples, 0)
    try:
        analytic_mean = unwrap_variable(dist.pyro_dist.analytic_mean(**dist_params))
        analytic_var = unwrap_variable(dist.pyro_dist.analytic_var(**dist_params))
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
    actual_support_non_vec = discrete_dist.pyro_dist.support(**discrete_dist.get_dist_params(SINGLE_TEST_DATUM_IDX))
    actual_support = discrete_dist.pyro_dist.support(**discrete_dist.get_dist_params(BATCH_TEST_DATA_IDX))
    assert_equal(actual_support.data, torch.Tensor(expected_support))
    assert_equal(actual_support_non_vec.data, torch.Tensor(expected_support_non_vec))


def get_batch_pdf_shape(dist, data, dist_params):
    d = dist.pyro_dist
    broadcasted_params = {}
    # TODO remove once #323 is resolved
    if dist.get_test_distribution_name() == 'Bernoulli':
        return d.batch_shape(data, **dist_params) + (1,)
    for p in dist_params:
        if dist_params[p].dim() < data.dim():
            broadcasted_params[p] = dist_params[p].expand_as(data)
        else:
            broadcasted_params[p] = dist_params[p]
    with xfail_if_not_implemented():
        return d.batch_shape(**broadcasted_params) + (1,)
