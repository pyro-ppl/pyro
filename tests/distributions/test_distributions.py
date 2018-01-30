from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch

from pyro.distributions.util import broadcast_shape
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal, xfail_if_not_implemented


def _unwrap_variable(x):
    return x.data.cpu().numpy()


def _log_prob_shape(dist, x_size=torch.Size(), **dist_params):
    event_dims = len(dist.event_shape(**dist_params))
    expected_shape = broadcast_shape(dist.shape(**dist_params), x_size, strict=True)
    if event_dims > 0:
        expected_shape = expected_shape[:-event_dims]
    if not expected_shape:
        expected_shape = (1,)
    return expected_shape

# Distribution tests - all distributions


def test_batch_log_prob(dist):
    if dist.scipy_arg_fn is None:
        pytest.skip('{}.log_pdf has no scipy equivalent'.format(dist.pyro_dist_obj.__name__))
    d = dist.pyro_dist
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        test_data = dist.get_test_data(idx)
        log_prob_sum_pyro = _unwrap_variable(d.log_prob(test_data, **dist_params)).sum()
        log_prob_sum_np = np.sum(dist.get_scipy_batch_logpdf(-1))
        assert_equal(log_prob_sum_pyro, log_prob_sum_np)


def test_sample_shape(dist):
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x_func = dist.pyro_dist.sample(**dist_params)
        x_obj = dist.pyro_dist_obj(**dist_params).sample()
        assert_equal(x_obj.size(), x_func.size())


def test_batch_log_prob_shape(dist):
    d = dist.pyro_dist
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x = dist.get_test_data(idx)
        with xfail_if_not_implemented():
            # Get log_prob shape after broadcasting.
            expected_shape = _log_prob_shape(d, x.size(), **dist_params)
            log_p_func = d.log_prob(x, **dist_params)
            log_p_obj = dist.pyro_dist_obj(**dist_params).log_prob(x)
            # assert that the functional and object forms return
            # the same log_prob values.
            assert_equal(log_p_func.size(), log_p_obj.size())
            assert log_p_func.size() == expected_shape


def test_log_prob_mask(dist):
    if dist.get_test_distribution_name() not in ('Normal', 'Bernoulli', 'Categorical', 'OneHotCategorical', 'Normal'):
        pytest.skip('Batch pdf masking not supported for the distribution.')
    d = dist.pyro_dist
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        x = dist.get_test_data(idx)
        with xfail_if_not_implemented():
            log_prob_shape = _log_prob_shape(d, **dist_params)
            log_prob_shape_broadcasted = _log_prob_shape(d, x.size(), **dist_params)
            zeros_mask = ng_zeros(1)  # should be broadcasted to data dims
            ones_mask = ng_ones(log_prob_shape)  # should be broadcasted to data dims
            half_mask = ng_ones(1) * 0.5
            batch_log_pdf = d.log_prob(x, **dist_params)
            log_prob_zeros_mask = d.log_prob(x, log_pdf_mask=zeros_mask, **dist_params)
            log_prob_ones_mask = d.log_prob(x, log_pdf_mask=ones_mask, **dist_params)
            log_prob_half_mask = d.log_prob(x, log_pdf_mask=half_mask, **dist_params)
            assert_equal(log_prob_ones_mask, batch_log_pdf)
            assert_equal(log_prob_zeros_mask, ng_zeros(log_prob_shape_broadcasted))
            assert_equal(log_prob_half_mask, 0.5 * batch_log_pdf)


def test_score_errors_event_dim_mismatch(dist):
    d = dist.pyro_dist
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        test_data_wrong_dims = ng_ones(d.shape(**dist_params) + (1,))
        if len(d.event_shape(**dist_params)) > 0:
            if dist.get_test_distribution_name() == 'MultivariateNormal':
                pytest.skip('MultivariateNormal does not do shape validation in log_prob.')
            with pytest.raises((ValueError, RuntimeError)):
                d.log_prob(test_data_wrong_dims, **dist_params)


def test_score_errors_non_broadcastable_data_shape(dist):
    d = dist.pyro_dist
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        shape = d.shape(**dist_params)
        non_broadcastable_shape = (shape[0] + 1,) + shape[1:]
        test_data_non_broadcastable = ng_ones(non_broadcastable_shape)
        with pytest.raises((ValueError, RuntimeError)):
            d.log_prob(test_data_non_broadcastable, **dist_params)


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
