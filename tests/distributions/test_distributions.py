from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch

from pyro.distributions import RandomPrimitive
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal, xfail_if_not_implemented


def unwrap_variable(x):
    return x.data.cpu().numpy()


# Distribution tests - all distributions

def test_log_pdf(dist):
    d = dist.pyro_dist
    for idx in dist.get_test_data_indices():
        dist_params = dist.get_dist_params(idx)
        test_data = dist.get_test_data(idx)
        pyro_log_pdf = unwrap_variable(d.log_pdf(test_data, **dist_params))[0]
        scipy_log_pdf = dist.get_scipy_logpdf(idx)
        assert_equal(pyro_log_pdf, scipy_log_pdf)


def test_batch_log_pdf(dist):
    d = dist.pyro_dist
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        test_data = dist.get_test_data(idx)
        logpdf_sum_pyro = unwrap_variable(torch.sum(d.batch_log_pdf(test_data, **dist_params)))[0]
        logpdf_sum_np = np.sum(dist.get_scipy_batch_logpdf(-1))
        assert_equal(logpdf_sum_pyro, logpdf_sum_np)


def test_shape(dist):
    d = dist.pyro_dist
    for idx in dist.get_test_data_indices():
        dist_params = dist.get_dist_params(idx)
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
            assert(x_func.size() == d.shape(x_func, **dist_params))


def test_batch_log_pdf_shape(dist):
    d = dist.pyro_dist
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x = dist.get_test_data(idx)
        with xfail_if_not_implemented():
            # Get batch pdf shape after broadcasting.
            expected_shape = d.batch_shape(x, **dist_params) + (1,)
            log_p_func = d.batch_log_pdf(x, **dist_params)
            log_p_obj = dist.pyro_dist_obj(**dist_params).batch_log_pdf(x)
            # assert that the functional and object forms return
            # the same batch pdf.
            assert_equal(log_p_func.size(), log_p_obj.size())
            assert log_p_func.size() == expected_shape


def test_batch_log_pdf_mask(dist):
    if dist.get_test_distribution_name() not in ('Normal', 'Bernoulli', 'Categorical'):
        pytest.skip('Batch pdf masking not supported for the distribution.')
    d = dist.pyro_dist
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x = dist.get_test_data(idx)
        with xfail_if_not_implemented():
            batch_pdf_shape = d.batch_shape(**dist_params) + (1,)
            batch_pdf_shape_broadcasted = d.batch_shape(x, **dist_params) + (1,)
            zeros_mask = ng_zeros(1)  # should be broadcasted to data dims
            ones_mask = ng_ones(batch_pdf_shape)  # should be broadcasted to data dims
            half_mask = ng_ones(1) * 0.5
            batch_log_pdf = d.batch_log_pdf(x, **dist_params)
            batch_log_pdf_zeros_mask = d.batch_log_pdf(x, log_pdf_mask=zeros_mask, **dist_params)
            batch_log_pdf_ones_mask = d.batch_log_pdf(x, log_pdf_mask=ones_mask, **dist_params)
            batch_log_pdf_half_mask = d.batch_log_pdf(x, log_pdf_mask=half_mask, **dist_params)
            assert_equal(batch_log_pdf_ones_mask, batch_log_pdf)
            assert_equal(batch_log_pdf_zeros_mask, ng_zeros(batch_pdf_shape_broadcasted))
            assert_equal(batch_log_pdf_half_mask, 0.5 * batch_log_pdf)


def test_mean_and_variance(dist):
    for idx in dist.get_test_data_indices():
        num_samples = dist.get_num_samples(idx)
        dist_params = dist.get_dist_params(idx)
        torch_samples = dist.get_samples(num_samples, **dist_params)
        sample_mean = torch_samples.float().mean(0)
        sample_var = torch_samples.float().var(0)
        try:
            analytic_mean = dist.pyro_dist.analytic_mean(**dist_params)
            analytic_var = dist.pyro_dist.analytic_var(**dist_params)
            assert_equal(sample_mean, analytic_mean, prec=dist.prec)
            assert_equal(sample_var, analytic_var, prec=dist.prec)
        except (NotImplementedError, ValueError):
            pytest.skip('analytic mean and variance are not available')


def test_score_errors_event_dim_mismatch(dist):
    d = dist.pyro_dist
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        test_data_wrong_dims = ng_ones(d.shape(**dist_params) + (1,))
        with pytest.raises(ValueError):
            d.batch_log_pdf(test_data_wrong_dims, **dist_params)


def test_score_errors_non_broadcastable_data_shape(dist):
    d = dist.pyro_dist
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        shape = d.shape(**dist_params)
        non_broadcastable_shape = (shape[0] + 1,) + shape[1:]
        test_data_non_broadcastable = ng_ones(non_broadcastable_shape)
        with pytest.raises(ValueError):
            d.batch_log_pdf(test_data_non_broadcastable, **dist_params)


# Distributions tests - discrete distributions

def test_enumerate_support(discrete_dist):
    expected_support = discrete_dist.expected_support
    expected_support_non_vec = discrete_dist.expected_support_non_vec
    if not expected_support:
        pytest.skip("enumerate_support not tested for distribution")
    actual_support_non_vec = discrete_dist.pyro_dist.enumerate_support(
        **discrete_dist.get_dist_params(0))
    actual_support = discrete_dist.pyro_dist.enumerate_support(
        **discrete_dist.get_dist_params(-1))
    assert_equal(actual_support.data, torch.Tensor(expected_support))
    assert_equal(actual_support_non_vec.data, torch.Tensor(expected_support_non_vec))


def get_batch_pdf_shape(dist, data, dist_params):
    d = dist.pyro_dist
    if isinstance(dist.pyro_dist, RandomPrimitive):
        return d.batch_shape(data, **dist_params) + (1,)
